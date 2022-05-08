import yaml
import os
from glob import glob
import sys

# import librosa
import numpy as np
import pandas as pd

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 100)

def find_feats(directory, pattern='**/*.pkl'):
    """Recursively finds all files matching the pattern."""
    return glob(os.path.join(directory, pattern), recursive=True)

def cnceleb_create(opts, pattern, mode):
    DB = pd.DataFrame()

    dir = opts[mode+'_feat_path'] + '/' + mode + '_logfbank_nfilt40'
    DB['filename'] = find_feats(dir, pattern) # filename

    DB['filename'] = DB['filename'].unique().tolist()
    DB['filename'] = DB['filename'].apply(lambda x: x.replace('\\', '/')) # normalize windows paths
    if mode == 'train':
        DB['speaker_id'] = DB['filename'].apply(lambda x: 'cnceleb_' + x.split('/')[-2]) # speaker folder name
        DB['device_id'] = DB['filename'].apply(lambda x: x.split('/')[-3]) # device name
    
    # Clean data
    # Discard the speakers with fewer than 5 utterances and the “advertisement” genre with fewer than 100 speakers in the training set.
    if mode == 'train':
        speaker_list = list(set(DB['speaker_id']))
        DB = DB[DB['device_id']!='advertisement']
        for speaker in speaker_list:
            if len(DB[DB['speaker_id']==speaker]) <=5:
                DB = DB[DB['speaker_id']!=speaker]

    DB.to_json('filelists/Cnceleb/cnceleb_' + mode +'.json')

if __name__ == '__main__':
    f = open('filelists/config.yaml', 'r')
    opts = yaml.load(f, Loader=yaml.CLoader)
    opts_cnceleb = opts['cnceleb']

    cnceleb_create(opts_cnceleb, '**/*.pkl', 'train')
    cnceleb_create(opts_cnceleb, '**/*.pkl', 'test')


    