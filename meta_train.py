import argparse
import yaml
import os
import random
import warnings
import numpy as np
import math

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

from utils import *
from Log import log

from generator.dataloader import baseGenerator
from generator.SR_Dataset import read_MFB_train as read_MFB
from model.feature_extractor import FeatureExtractor
from model.gener_gmlp import gMLPGener as gMLPGener
from loss.LossClassification import ClsLoss
from loss.LossMeta import MetaLoss

parser = argparse.ArgumentParser()
parser.add_argument('--n_folder', type=int, default=1, help='Number of folder.')

parser.add_argument('--generalization', type=str, default='gmlp', help='The name of the generalized network used.')

parser.add_argument('--dataset', type=str, default='cnceleb', help='Dataset name. ex: cnceleb')
parser.add_argument('--domain', type=str, default='g1,g2,g3,g4', help='Selected domain. ex: g1,g2')

parser.add_argument('--loss_type', type=str, default='metaloss', help='metaloss.')
parser.add_argument('--use_GC', type=str2bool, default=True, help='Use classification logit.')

parser.add_argument('--use_checkpoint', type=str2bool, default=False, help='Use checkpoint.')
parser.add_argument('--cp_num', type=int, default=0, help='Number of checkpoint.')
parser.add_argument('--other', type=str, default='', help='Extract information.')

parser.add_argument('--n_shot', type=int, default=1, help='P: Number of support set per class.')
parser.add_argument('--n_query', type=int, default=2, help='Q: Number of query set per class.')
parser.add_argument('--nb_class_train', type=int, default=32, help='Number of way for training episode, similar to batchsize.')

args = parser.parse_args()

# Load fixed parameters
f = open('config.yaml', 'r')
opts = yaml.load(f, Loader=yaml.CLoader)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(opts['base']['gpu'])
cuda = torch.device('cuda')
device_ids = list(range(len(str(opts['base']['gpu']).split(','))))
log_dir = 'saved_model/meta_' + args.generalization + '_' + args.dataset + '_' + args.domain + '_' + args.loss_type + '_' + args.other + '_' + str(args.n_folder).zfill(3)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Save all parameters of args
save_parser(log_dir, args)

def main():
    seed = opts['base']['seed']
    deterministic = True

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.deterministic = deterministic
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    log.log('Initialized ' + args.loss_type)
    log.log('Dataset: ' + args.dataset + '/ Domain: ' + args.domain)
    log.log('Generalized Net: ' + args.generalization)
    log.log(torch.cuda.is_available())

    if args.use_checkpoint: start = args.cp_num + 1
    else: start = 0  # Start epoch
    n_epochs = opts['base']['max_epoch'] - start  # How many epochs?

    # Load dataset
    trainset = baseGenerator(opts['data'], args, read_MFB,
                            args.nb_class_train, nb_samples_per_class=args.n_shot + args.n_query)
    metagenerator = DataLoader(trainset, batch_size=args.nb_class_train, shuffle=True,
                                collate_fn=trainset.collate_fn, num_workers=16)
    n_data, n_classes = trainset.n_data, trainset.n_classes
    n_episode = math.floor(n_data / ((args.n_shot + args.n_query) * args.nb_class_train))
    trainset.max_iter = n_episode * (n_epochs-args.cp_num)

    # Generate model and optimizer
    feature_extractor = FeatureExtractor(backbone='resnet18', maml=True)
    if args.generalization == 'gmlp':
        generalized_net = gMLPGener(dim = 3, depth = 3, seq_len = 256, prob_survival = 0.5)

    if args.use_checkpoint:
        feature_extractor, generalized_net = load_model(feature_extractor, generalized_net, log_dir, args.cp_num)
    
    # 损失函数选择
    opts_model = opts['model']
    if args.loss_type == 'metaloss':
        objective = MetaLoss(opts_model['embedding_size'], n_classes, args.use_GC)
        lr = opts['meta']['lr'][0]
    else:
        raise NotImplementedError("Other loss function has not been implemented yet!")
    if args.use_checkpoint:
        objective = load_object(objective)
    
    # 优化器定义
    mtr_param = [{'params': feature_extractor.parameters()}, {'params': objective.parameters()}]
    mte_param = [{'params': generalized_net.parameters()}]

    mtr_optim = optim.SGD(mtr_param, lr=lr, momentum=0.9, weight_decay = 0.00001, nesterov=True, dampening=0)
    mte_optim = optim.SGD(mte_param, lr=lr*0.1, momentum=0.9, weight_decay = 0.00001, nesterov=True, dampening=0)
    if args.use_checkpoint:
        mtr_optim = load_optim(mtr_optim, 'mtr_optim')
        mte_optim = load_optim(mte_optim, 'mte_optim')

    mtr_sch = optim.lr_scheduler.StepLR(mtr_optim, step_size=opts['meta']['lr_decay_step'], gamma=0.2)
    mte_sch = optim.lr_scheduler.StepLR(mte_optim, step_size=opts['meta']['lr_decay_step'], gamma=0.2)

    feature_extractor = torch.nn.DataParallel(feature_extractor.to(cuda), device_ids=device_ids)
    generalized_net = torch.nn.DataParallel(generalized_net.to(cuda), device_ids=device_ids)
    objective = torch.nn.DataParallel(objective.to(cuda), device_ids=device_ids)

    train(metagenerator, feature_extractor, generalized_net, objective, mtr_optim, mte_optim, n_episode, log_dir, mtr_sch, mte_sch)

def train(metagenerator, feature_extractor, generalized_net, objective, mtr_optim, mte_optim, n_episode, log_dir, mtr_sch, mte_sch):
    log_interval = int(n_episode / 5)
    avg_train_losses = []

    for epoch in range(opts['base']['max_epoch'] - args.cp_num):
        feature_extractor.train()
        generalized_net.train()
        objective.train()

        losses_mtr = AverageMeter()
        losses_e_mtr = AverageMeter()
        losses_g_mtr = AverageMeter()
        accuracy_e_mtr = AverageMeter()
        accuracy_g_mtr = AverageMeter()

        losses_mte = AverageMeter()
        losses_e_mte = AverageMeter()
        losses_g_mte = AverageMeter()
        accuracy_e_mte = AverageMeter()
        accuracy_g_mte = AverageMeter()

        mtr_optim.zero_grad()
        mte_optim.zero_grad()

        for t, (data) in enumerate(metagenerator):
            mtr_inputs, mtr_labels, mte_inputs, mte_labels = data

            feature_extractor.train()
            generalized_net.train()
            objective.train()

            if args.loss_type == 'metaloss':
                '''
                1) Meta-Train Training.
                '''
                for weight in split_model_parameters(feature_extractor):
                    weight.fast = None
                
                targets_e_mtr = tuple([i for i in range(args.nb_class_train)]) * (args.n_query)
                targets_e_mtr = torch.tensor(targets_e_mtr, dtype=torch.long)
                support_mtr, query_mtr = split_support_query(mtr_inputs, args)
                support_mtr = support_mtr - torch.mean(support_mtr, dim=3, keepdim=True)
                query_mtr = query_mtr - torch.mean(query_mtr, dim=3, keepdim=True)
                support_mtr, query_mtr, mtr_labels, targets_e_mtr = support_mtr.to(cuda), query_mtr.to(cuda), mtr_labels.to(cuda), targets_e_mtr.to(cuda)

                support_mtr = feature_extractor(support_mtr, 'mte')
                query_mtr = feature_extractor(query_mtr, 'mte')
                loss_mtr, loss_e_mtr, loss_g_mtr, acc_e_mtr, acc_g_mtr =\
                    objective(support_mtr, query_mtr, mtr_labels, targets_e_mtr, generalized_net, mode='mtr')

                meta_grad = torch.autograd.grad(loss_mtr, split_model_parameters(feature_extractor), create_graph=True)
                for k, weight in enumerate(split_model_parameters(feature_extractor)):
                    weight.fast = weight - mtr_optim.param_groups[0]['lr'] * meta_grad[k]

                losses_mtr.update(loss_mtr.item(), query_mtr.size(0))
                losses_e_mtr.update(loss_e_mtr.item(), query_mtr.size(0))
                losses_g_mtr.update(loss_g_mtr.item(), mtr_inputs.size(0))
                accuracy_e_mtr.update(acc_e_mtr * 100, query_mtr.size(0))
                accuracy_g_mtr.update(acc_g_mtr * 100, mtr_inputs.size(0))

                
                # ===========================================================
                '''
                2) Meta-Test Training.
                '''
                feature_extractor.eval()
                targets_e_mte = tuple([i for i in range(args.nb_class_train)]) * (args.n_query)
                targets_e_mte = torch.tensor(targets_e_mte, dtype=torch.long)
                support_mte, query_mte = split_support_query(mte_inputs, args)
                support_mte = support_mte - torch.mean(support_mte, dim=3, keepdim=True)
                query_mte = query_mte - torch.mean(query_mte, dim=3, keepdim=True)
                support_mte, query_mte, mte_labels, targets_e_mte = support_mte.to(cuda), query_mte.to(cuda), mte_labels.to(cuda), targets_e_mte.to(cuda)

                support_mte = feature_extractor(support_mte, 'mte')
                query_mte = feature_extractor(query_mte, 'mte')
                loss_mte, loss_e_mte, loss_g_mte, acc_e_mte, acc_g_mte =\
                    objective(support_mte, query_mte, mte_labels, targets_e_mte, generalized_net, mode='mte')

                # ===========================================================
                '''
                3) Summary.
                '''
                mtr_optim.zero_grad()
                loss_mtr.backward(retain_graph=True)
                mtr_optim.step()

                mte_optim.zero_grad()
                loss_mte.backward()
                mte_optim.step()

                losses_mte.update(loss_mte.item(), query_mte.size(0))
                losses_e_mte.update(loss_e_mte.item(), query_mte.size(0))
                losses_g_mte.update(loss_g_mte.item(), mte_inputs.size(0))
                accuracy_e_mte.update(acc_e_mte * 100, query_mte.size(0))
                accuracy_g_mte.update(acc_g_mte * 100, mte_inputs.size(0))

                # ===========================================================

                # episode number in epoch
                ith_episode = t % n_episode
                if ith_episode % log_interval == 0:
                    log.log(
                        'Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\t'
                        'Loss {loss_mtr.avg:.4f}|{loss_mte.avg:.4f} '
                        '(loss_e: {loss_e_mtr.avg:.4f}|{loss_e_mte.avg:.4f} / loss_g: {loss_g_mtr.avg:.4f}|{loss_g_mte.avg:.4f})\t'
                        'Acc e / g {acc_e_mtr.avg:.4f}|{acc_e_mte.avg:.4f} / {acc_g_mtr.avg:.4f}|{acc_g_mte.avg:.4f}'.format(
                        epoch, ith_episode, n_episode, 100. * ith_episode / n_episode,
                        loss_mtr=losses_mtr, loss_mte=losses_mte,
                        loss_e_mtr=losses_e_mtr, loss_e_mte=losses_e_mte,
                        loss_g_mtr=losses_g_mtr, loss_g_mte=losses_g_mte,
                        acc_e_mtr=accuracy_e_mtr, acc_e_mte=accuracy_e_mte,
                        acc_g_mtr=accuracy_g_mtr, acc_g_mte=accuracy_g_mte))
                    log.log('learning rate: ' + str(mtr_optim.param_groups[0]['lr']))
                

        # calculate average loss over an epoch
        mtr_sch.step()
        mte_sch.step()
        # scheduler.step(losses_mte.avg)
        avg_train_losses.append(losses_mte.avg)
          
        if epoch % opts['train']['save_step'] == 0 and epoch != 0:
        # if epoch % opts['train']['save_step'] == 0:
            torch.save({'epoch': epoch + 1, 
                        'featurenet': feature_extractor.state_dict(),
                        'generalizednet': generalized_net.state_dict(),
                        'mtr_optim': mtr_optim.state_dict(),
                        'mte_optim': mte_optim.state_dict(),
                        'loss': objective.state_dict()},
                       '{}/checkpoint_{}.pth'.format(log_dir, str(epoch).zfill(3)))

        # Test the model at intervals of several epochs
        # if epoch % opts['train']['test_step'] == 0 and epoch != 0:
        if epoch % opts['train']['test_step'] == 0:
            log.log('<=====================>')
            log.log('Test current model performance')
            train_task(feature_extractor, generalized_net, args, opts, epoch)
            log.log('<=====================>')


# Import and load models
def load_model(featurenet, generalizednet, log_dir, cp_num):
    log.log('=> loading checkpoint')
    checkpoint = torch.load(log_dir + '/checkpoint_' + str(cp_num).zfill(3) + '.pth')
    # create new OrderedDict that does not contain `module.`
    featurenet.load_state_dict(checkpoint['featurenet'])
    generalizednet.load_state_dict(checkpoint['generalizednet'])

    return featurenet, generalizednet

def load_optim(optimizer, optim_name):
    checkpoint = torch.load(log_dir + '/checkpoint_' + str(args.cp_num).zfill(3) + '.pth')
    optimizer.load_state_dict(checkpoint[optim_name])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    return optimizer

def load_object(objective):
    checkpoint = torch.load(log_dir + '/checkpoint_' + str(args.cp_num).zfill(3) + '.pth')
    # create new OrderedDict that does not contain `module.`
    objective.load_state_dict(checkpoint['loss'])
    return objective

def cycle(iterable):
  while True:
    for x in iterable:
      yield x

def split_model_parameters(model):
    model_params = []
    for n, p in model.named_parameters():
        model_params.append(p)
    return model_params


def train_task(feature_extractor, generalized_net, args, opts, epoch):
    feature_extractor.eval()
    generalized_net.eval()

    if ',' in args.dataset:
        select_dataset = args.dataset.split(',')[0]
    else:
        select_dataset = args.dataset
    train_task, train_files = get_sample(select_dataset, opts)

    dict_embeddings = {}
    total_len = len(train_files)
    with torch.no_grad():
        for i in range(len(train_files)):
            if train_files[i] == '':
                continue
            tmp_filename = train_files[i]
            enroll_embedding, _ = get_meta_d_vector(tmp_filename, feature_extractor, generalized_net, opts['train'][select_dataset+'_test_path']) 
            # key = tmp_filename.split(os.sep)[-1]  # ex) 'id10042/6D67SnCYY34/00001.pkl'
            # key = os.path.splitext(key)[0] + '.wav'  # ex) 'id10042/6D67SnCYY34/00001.wav'
            key = tmp_filename.replace('.pkl', '.wav')
            dict_embeddings[key] = enroll_embedding
            # log.log("[%s/%s] Embedding for \"%s\" is saved" % (str(i).zfill(len(str(total_len))), total_len, key))

    _ = perform_verification(train_task, dict_embeddings, select_dataset, epoch)

def perform_verification(lines, dict_embeddings, select_dataset, epoch):
    # Perform speaker verification using veri_test.txt

    score_list = []
    label_list = []
    num = 0

    for line in lines:
        if not line: break

        enroll_filename, test_filename, label = line.split(' ')[0], line.split(' ')[1], int(line.split(' ')[2].replace("\n", ""))
        with torch.no_grad():
            # Get embeddings from dictionary
            # Get enroll embedding and test embedding according to embedding dict
            enroll_embedding = dict_embeddings[enroll_filename]
            test_embedding = dict_embeddings[test_filename]

            # Calculate cosine distance
            score = F.cosine_similarity(enroll_embedding, test_embedding)
            score = score.data.cpu().numpy()[0]
            del enroll_embedding
            del test_embedding

        score_list.append(score)
        label_list.append(label)
        num += 1

    eer, eer_threshold = get_eer(score_list, label_list, args, epoch)
    return eer

            
if __name__ == '__main__':
    main()