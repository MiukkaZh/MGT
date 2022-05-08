import yaml
import os
from glob import glob
import pandas as pd
import numpy as np
import sys
import soundfile as sf
from python_speech_features import mfcc, fbank, logfbank, delta
import pickle
from Log import log


np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 100)

def find_wavs(directory, pattern='*/*.wav'):
    """Recursively finds all waves matching the pattern."""
    return glob(os.path.join(directory, pattern), recursive=True)

def read_cnceleb_structure(directory, data_type):
    cnceleb = pd.DataFrame()

    if data_type == 'wavs':
        cnceleb['filename'] = find_wavs(directory)
    elif data_type == 'test':
        cnceleb['filename'] = find_wavs(directory, pattern='*/*.wav')
    else:
        raise NotImplementedError
    # Normalize the path format
    cnceleb['filename'] = cnceleb['filename'].apply(lambda x: x.replace('\\', '/')) # normalize windows paths
    num_speakers = len(os.listdir(directory))
    log.log('Found {} files with {} different speakers.'.format(str(len(cnceleb)).zfill(7), str(num_speakers).zfill(5)))
    log.log(cnceleb.head(10))
    
    return cnceleb


def normalize(feat):
    return (feat - feat.mean(axis = 0)) / (feat.std(axis = 0) + 2e-12)

def delta(feat, order = 2):
    if order == 2:
        feat_d1 = delta(feat, N = 1)
        feat_d2 = delta(feat, N = 2)
        feat = np.hstack([feat, feat_d1, feat_d2])
    elif order == 1:
        feat_d1 = delta(feat, N = 1)
        feat = np.hstack([feat, feat_d1])
    return feat

def convert_wav_to_MFB_name(filename, opts, mode):
    """
    Converts the wav dir (in DB folder) to feat dir(in feat folder)
    """
    data_type = filename.split('/')[-4]
    filename_only = filename.split('/')[-1].replace('.wav','.pkl') # ex) 00001.pkl (pickle format)
    uri_folder = filename.split('/')[-1].split('-')[-3]            # ex) sing
    speaker_folder = filename.split('/')[-2]                       # ex) id00978

    if uri_folder == 'moives':
        uri_folder = 'movie'

    if opts['feat_type'] == 'logfbank':
        feature_type = 'logfbank'
    elif opts['feat_type'] == 'fbank':
        feature_type = 'fbank'

    if mode == 'train':
        # 示例) feat/train_logfbank_nfilt40
        if opts['delta']:
            root_folder = 'train_' + feature_type + '_nfilt' + str(opts['filter_fbank']) + '_del2'
        else:
            root_folder = 'train_' + feature_type + '_nfilt' + str(opts['filter_fbank'])

        feat_only_dir = opts['train_feat_path']
        output_foldername = os.path.join(feat_only_dir, root_folder, uri_folder, speaker_folder)
        
    elif mode == 'test':
        # 示例) feat/test_logfbank_nfilt40
        if opts['delta']:
            root_folder = 'test_' + feature_type + '_nfilt' + str(opts['filter_fbank']) + '_del2'
        else:
            root_folder = 'test_' + feature_type + '_nfilt' + str(opts['filter_fbank'])
        output_foldername = os.path.join(opts['test_feat_path'], root_folder, uri_folder, speaker_folder)
        
    output_filename = os.path.join(output_foldername, filename_only)

    return output_foldername, output_filename

def extract_MFB(filename, opts, mode):
    audio, sr = sf.read(filename, dtype='float32', always_2d=True)
    audio = audio[:, 0]

    if opts['feat_type'] == 'logfbank':
        features = logfbank(audio, opts['rate'], winlen = opts['win_len'], winstep = opts['win_shift'], nfilt = opts['filter_fbank'])
    elif opts['feat_type'] == 'fbank':
        features, _ = fbank(audio, opts['rate'], winlen = opts['win_len'], winstep = opts['win_shift'], nfilt = opts['filter_fbank'])
    else:
        raise NotImplementedError("Other features are not implemented!")
    
    if opts['normalize']:
        features = normalize(features)
    if opts['delta']:
        features = delta(features)
    
    total_features = features
    if mode == 'train':
        # Convert .wav to .pkl format
        speaker_folder = filename.split('/')[-2]
        output_foldername, output_filename = convert_wav_to_MFB_name(filename, opts, mode=mode)
        speaker_label = speaker_folder # set label as a folder name (recommended). Convert this to speaker index when training
        feat_and_label = {'feat':total_features, 'label':speaker_label}
        accept = int(speaker_label[2:]) not in range(800, 1000)

    elif mode == 'test':
        root_folder = 'test_' + opts['feat_type'] + '_nfilt' + str(opts['filter_fbank'])
        filename_only = filename.split('/')[-1].replace('.wav','.pkl')
        output_foldername = os.path.join(opts['test_feat_path'], root_folder)
        speaker_label = filename.split('/')[-1].split('-')[0]
        output_foldername = os.path.join(output_foldername, filename.split('/')[-2])
        feat_and_label = {'feat': total_features, 'label':speaker_label}
        output_filename = os.path.join(output_foldername, filename_only)
        accept = int(speaker_label[2:]) in range(800, 1000)

    if accept:
        if not os.path.exists(output_foldername):
            os.makedirs(output_foldername)

        if os.path.isfile(output_filename) == 1:
            log.log("\"" + '/'.join(output_filename.split('/')[-3:]) + "\"" + " file already extracted!")
        else:
            with open(output_filename, 'wb') as fp:
                pickle.dump(feat_and_label, fp)

        return True
    else:
        return False


class mode_error(Exception):
    def __str__(self):
        return "Wrong mode (type 'train' or 'test')"

def feat_extraction(opts, mode):
    if mode == 'train':
        DB = read_cnceleb_structure(opts['train_audio_path'], data_type='wavs')
    else:
        DB = read_cnceleb_structure(opts['test_audio_path'], data_type='test')
    
    if (mode != 'train') and (mode != 'test'):
      raise mode_error
    count = 0
    
    # Extract features for each file
    for i in range(len(DB)):
        accept = extract_MFB(DB['filename'][i], opts, mode=mode)
        if accept:
            count = count + 1
            filename = DB['filename'][i]
            log.log("feature extraction (%s DB). step : %d, file : \"%s\"" %(mode, count, '/'.join(filename.split('/')[-2:])))

    log.log("-"*20 + " Feature extraction done " + "-"*20)


if __name__ == '__main__':
    f = open('filelists/config.yaml', 'r')
    opts = yaml.load(f, Loader=yaml.CLoader)['cnceleb']
    dataset = opts['train_audio_path'].split(', ')

    for index in range(len(dataset)):
        opts['train_audio_path'] = dataset[index]
        feat_extraction(opts, mode='train')
    feat_extraction(opts, mode='test')
