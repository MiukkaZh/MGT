from torch.nn.modules import activation
from Log import log
import torch
from generator.SR_Dataset import *
from torch.autograd import Variable
import os
from sklearn.metrics import roc_curve

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

def save_parser(log_dir, parser):
    with open(log_dir+'/parser.txt', 'w') as f:
        for k,v in vars(parser).items():
            f.write(k + ': ' + str(v) + '\n')

    log.log('Parser has been saved!')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def split_support_query(inputs, args):
    B, C, Fr, T = inputs.size()
    inputs = inputs.reshape(args.n_shot + args.n_query, args.nb_class_train, C, Fr, T)
    support = inputs[:args.n_shot].reshape(-1, C, Fr, T)
    query = inputs[args.n_shot:].reshape(-1, C, Fr, T)

    return support, query

def split_label(labels, args):
    support_label = labels[:args.n_shot*args.nb_class_train]
    query_label = labels[args.n_shot*args.nb_class_train:]

    return support_label, query_label
    
    
def normalize(feat):
    return (feat - feat.mean(axis = 0)) / (feat.std(axis = 0) + 2e-12)

def get_sample(select_dataset, opts):
    with open(opts['train'][select_dataset + '_train_task'], 'r') as f:
        train_task = f.read().split('\n')
    with open(opts['train'][select_dataset + '_train_files'], 'r') as f:
        train_files = f.read().split('\n')
    
    return train_task, train_files

def get_meta_d_vector(filename, feature_extractor, generalized_net, root_path):
    # 导入 feat and label
    input, label = test_input_load(filename, root_path)
    label = torch.tensor([1]).cuda()

    input = normalize(input)
    TT = ToTensorTestInput()  # torch tensor:(1, n_dims, n_frames)
    input = TT(input)  # size : (n_frames, 1, n_filter, T)
    input = Variable(input)
    with torch.no_grad():
        cuda = torch.device('cuda')
        input = input.to(cuda)
        label = label.to(cuda)

        #scoring function is cosine similarity so, you don't need to normalization
        feature = feature_extractor(input)
        activation = generalized_net(feature)

    return activation, label

def get_d_vector(filename, model, root_path):
    # Load feat and label
    input, label = test_input_load(filename, root_path)
    label = torch.tensor([1]).cuda()

    input = normalize(input)
    TT = ToTensorTestInput()  # torch tensor:(1, n_dims, n_frames)
    input = TT(input)  # size : (n_frames, 1, n_filter, T)
    input = Variable(input)
    with torch.no_grad():
        cuda = torch.device('cuda')
        input = input.to(cuda)
        label = label.to(cuda)

        activation = model(input) #scoring function is cosine similarity so, you don't need to normalization

    return activation, label

def test_input_load(filename, root_path):
    # Loading pkl files
    feat_name = filename.replace('.wav', '.pkl')
    mod_filename = os.path.join(root_path, feat_name)

    file_loader = read_MFB
    input, label = file_loader(mod_filename)  # input size :(n_frames, dim), label:'id10309'

    return input, label

def get_eer(score_list, label_list, args, epoch=0):
    fpr, tpr, threshold = roc_curve(label_list, score_list, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    intersection = abs(1 - tpr - fpr)
    DCF2 = 100 * (0.01 * (1 - tpr) + 0.99 * fpr)
    DCF3 = 1000 * (0.001 * (1 - tpr) + 0.999 * fpr)
    log.log("Epoch=%d  EER= %.2f  Thres= %0.5f" % (
    epoch, 100 * fpr[np.argmin(intersection)], eer_threshold))

    return eer, eer_threshold
    