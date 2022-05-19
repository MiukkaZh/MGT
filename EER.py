import argparse
import warnings
import yaml
import os
import time

import torch
import torch.nn.functional as F

from model.feature_extractor import FeatureExtractor
from model.gener_gmlp import gMLPGener as gMLPGener
from Log import log
from utils import *


warnings.filterwarnings("ignore", message="numpy.dtype size changed")

parser = argparse.ArgumentParser()
parser.add_argument('--n_folder', type=int, default=0, help='Number of folder.')
parser.add_argument('--cp_num', type=int, default=15, help='Number of checkpoint.')

parser.add_argument('--model_type', type=str, default='meta', help='meta / baseline')
parser.add_argument('--data_type', type=str, default='cnceleb', help='cnceleb')
parser.add_argument('--loss_type', type=str, default='metaloss', help='metaloss.')
parser.add_argument('--dataset', type=str, default='cnceleb', help='Dataset name. ex: cnceleb,himia')
parser.add_argument('--domain', type=str, default='g1,g2,g3,g4', help='Selected domain. ex: movie,play')
parser.add_argument('--generalization', type=str, default='gmlp', help='The name of the generalized network used.')

parser.add_argument('--veri_test_dir', type=str, default='trials/cnceleb/fix_cnceleb_trials.txt', help='Pairs File Root.')
parser.add_argument('--other', type=str, default='', help='Extract information.')

args = parser.parse_args()

f = open('config.yaml', 'r')
opts = yaml.load(f, Loader=yaml.CLoader)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(opts['test']['gpu'])

cuda = torch.device('cuda')
device_ids = list(range(len(str(opts['base']['gpu']).split(','))))
if args.model_type == 'meta':
    log_dir = 'saved_model/meta_' + args.generalization + '_' + args.dataset + '_' + args.domain + '_' + args.loss_type + '_' + args.other + '_' + str(args.n_folder).zfill(3)
elif args.model_type == 'baseline':
    log_dir = 'saved_model/baseline_' + args.dataset + '_' + args.domain + '_' + args.loss_type + '_' + args.other + '_' + str(args.n_folder).zfill(3)

def main():
    featurenet, generalizednet = load_model(device_ids, log_dir, args.cp_num)

    # Enroll and test
    to_start = time.time()
    test_trials, test_files = get_sample(args, opts)
    dict_embeddings = {}
    total_len = len(test_files)
    with torch.no_grad():
        for i in range(len(test_files)):
            if test_files[i] == '':
                continue
            tmp_filename = test_files[i]
            enroll_embedding, _ = get_meta_d_vector(tmp_filename, featurenet, generalizednet, opts['test'][args.data_type.split('_')[0]+'_test_path'])
            key = tmp_filename.replace('.pkl', '.wav')
            dict_embeddings[key] = enroll_embedding
            log.log("[%s/%s] Embedding for \"%s\" is saved" % (str(i).zfill(len(str(total_len))), total_len, key))
    enroll_time = time.time() - to_start

    log.log(log_dir)
    verification_start = time.time()
    _ = perform_verification(test_trials, dict_embeddings, args.data_type)
    log.log(args.veri_test_dir)
    tot_end = time.time()
    verification_time = tot_end - verification_start
    meta_trials = os.listdir(opts['test'][args.data_type.split('_')[0]+'_meta'])
    for meta_trial in meta_trials:
        with open(opts['test'][args.data_type.split('_')[0]+'_meta']+'/'+meta_trial, 'r') as f:
            test_trials = f.read().split('\n')
        log.log(meta_trial)
        _ = perform_verification(test_trials, dict_embeddings, args.data_type)
    log.log("Time elapsed for enroll : %0.1fs" % enroll_time)
    log.log("Time elapsed for verification : %0.1fs" % verification_time)
    log.log("Total elapsed time : %0.1fs" % (tot_end - to_start))
    log.log("==============================================================================")
    log.log("==============================================================================")
    log.log("==============================================================================")
    log.log("==============================================================================")
    log.log("==============================================================================")
    log.log("==============================================================================")


def load_model(device_ids, log_dir, cp_num):
    feature_extractor = FeatureExtractor(backbone='resnet18', maml=True)
    if args.generalization == 'gmlp':
        generalized_net = gMLPGener(dim = 3, depth = 3, seq_len = 256)

    feature_extractor = torch.nn.DataParallel(feature_extractor.to(cuda), device_ids=device_ids)
    generalized_net = torch.nn.DataParallel(generalized_net.to(cuda), device_ids=device_ids)
    log.log('=> loading checkpoint')
    # load pre-trained parameters
    checkpoint = torch.load(log_dir + '/checkpoint_' + str(cp_num).zfill(3) + '.pth')
    feature_extractor.load_state_dict(checkpoint['featurenet'])
    generalized_net.load_state_dict(checkpoint['generalizednet'])
    feature_extractor.eval()
    generalized_net.eval()
    return feature_extractor, generalized_net

def get_sample(args, opts):
    with open(args.veri_test_dir, 'r') as f:
        test_trials = f.read().split('\n')
    with open(opts['test'][args.data_type + '_test_files'], 'r') as f:
        test_files = f.read().split('\n')
    
    return test_trials, test_files

def perform_verification(lines, dict_embeddings, select_dataset):
    # Perform speaker verification using veri_test.txt

    score_list = []
    label_list = []
    num = 0

    for line in lines:
        if not line: break

        if select_dataset == 'vox':
            label, enroll_filename, test_filename = int(line.split(' ')[0]), line.split(' ')[1], line.split(' ')[2].replace("\n", "")
        else:
            enroll_filename, test_filename, label = line.split(' ')[0], line.split(' ')[1], int(line.split(' ')[2].replace("\n", ""))
        with torch.no_grad():
            # Get embeddings from dictionary
            enroll_embedding = dict_embeddings[enroll_filename]
            test_embedding = dict_embeddings[test_filename]

            score = F.cosine_similarity(enroll_embedding, test_embedding)
            score = score.data.cpu().numpy()[0]
            del enroll_embedding
            del test_embedding

        score_list.append(score)
        label_list.append(label)
        num += 1

    eer, eer_threshold = get_eer(score_list, label_list, args)
    return eer

if __name__ == '__main__':
    main()