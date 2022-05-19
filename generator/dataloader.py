import numpy as np
import pandas as pd
import random
import torch
import torchvision.transforms as transforms
from generator.SR_Dataset import TruncatedInputfromMFB, ToTensorInput
from Log import log
from torch.utils.data import Dataset
import math


class baseGenerator(Dataset):
    '''
        opts: Options for database file location
        args: The number of selected classes, the number of samples per class
    '''
    def __init__(self, opts, args, file_loader, nb_classes=100, nb_samples_per_class=3, max_epoch=100, xp=np):
        super(baseGenerator, self).__init__()

        self.opts = opts
        self.nb_classes = nb_classes
        self.nb_samples_per_class = nb_samples_per_class
        self.max_epoch = max_epoch
        self.xp = xp
        self.args = args
        self.num_iter = 0
        self.data, self.n_classes, self.n_data = self._load_data()
        self.file_loader = file_loader
        self.transform = transforms.Compose([
            TruncatedInputfromMFB(opts=self.opts),  # numpy array:(1, n_frames, n_dims)
            ToTensorInput()  # torch tensor:(1, n_dims, n_frames)
        ])
        self.length = math.floor(self.n_data / self.nb_samples_per_class)
    
    def _load_data(self):
        '''
            ex: 0: [1.pkl, 2.pkl ...]
        '''
        # Loading CN-Celeb
        cn_DB = pd.DataFrame()

        if self.args.dataset != '':

            if self.args.domain:
                selec_DB = pd.read_json(self.opts[self.args.dataset+'_database'])
                if self.args.dataset == 'cnceleb':
                    selec_domains = self.args.domain.replace('g1', self.opts['g1'])
                    selec_domains = selec_domains.replace('g2', self.opts['g2'])
                    selec_domains = selec_domains.replace('g3', self.opts['g3'])
                    selec_domains = selec_domains.replace('g4', self.opts['g4'])
                    selec_domains = selec_domains.split(',')
                else:
                    selec_domains = self.args.domain.split(',')
                selec_DB = selec_DB[selec_DB['device_id'].isin(selec_domains)]
                DB = pd.concat([cn_DB, selec_DB], ignore_index=True)

            else:
                selec_datasets = self.args.dataset.split(',')
                DB = cn_DB
                for selec_dataset in selec_datasets:
                    selec_DB = pd.read_json(self.opts[selec_dataset+'_split_database'])
                    DB = pd.concat([DB, selec_DB], ignore_index=True)
        else:
            DB = cn_DB
        
        speaker_list = sorted(set(DB['speaker_id']))
        spk_to_idx = {spk: i for i, spk in enumerate(speaker_list)}
        DB['labels'] = DB['speaker_id'].apply(lambda x: spk_to_idx[x])
        num_speakers = len(DB['speaker_id'].unique())
        log.log('Found {} different speakers.'.format(str(num_speakers).zfill(5)))

        data = {key: np.array(DB.loc[DB['labels']==key]['filename']) for key in range(num_speakers)}

        return data, num_speakers, len(DB)
    
    def collate_fn(self, batch):
        picture_list = sorted(set(self.data.keys()))
        mtr_sampled_characters = random.sample(self.data.keys(), self.nb_classes)
        mtr_labels_and_images = []
        for (k, char) in enumerate(mtr_sampled_characters):
            label = char
            _imgs = self.data[char]
            if len(_imgs) >= self.nb_samples_per_class:
                _ind = random.sample(range(len(_imgs)), self.nb_samples_per_class)

            else:
                _ind = random.choices(population=range(len(_imgs)), k=self.nb_samples_per_class)
            mtr_labels_and_images.extend([(label, self.transform(self.file_loader(_imgs[i]))) for i in _ind])
        mtr_arg_labels_and_images = []
        for i in range(self.nb_samples_per_class):
            for j in range(self.nb_classes):
                mtr_arg_labels_and_images.extend([mtr_labels_and_images[i+j*self.nb_samples_per_class]])

        mtr_labels, mtr_images = zip(*mtr_arg_labels_and_images)
        mtr_images = torch.stack(mtr_images, dim=0)
        mtr_labels = torch.tensor(mtr_labels, dtype=torch.long)

        picture_list = sorted(set(self.data.keys()))
        mte_sampled_characters = random.sample(self.data.keys(), self.nb_classes)
        mte_labels_and_images = []
        for (k, char) in enumerate(mte_sampled_characters):
            label = char
            _imgs = self.data[char]
            if len(_imgs) >= self.nb_samples_per_class:
                _ind = random.sample(range(len(_imgs)), self.nb_samples_per_class)

            else:
                _ind = random.choices(population=range(len(_imgs)), k=self.nb_samples_per_class)
            mte_labels_and_images.extend([(label, self.transform(self.file_loader(_imgs[i]))) for i in _ind])
        mte_arg_labels_and_images = []
        for i in range(self.nb_samples_per_class):
            for j in range(self.nb_classes):
                mte_arg_labels_and_images.extend([mte_labels_and_images[i+j*self.nb_samples_per_class]])

        mte_labels, mte_images = zip(*mte_arg_labels_and_images)
        mte_images = torch.stack(mte_images, dim=0)
        mte_labels = torch.tensor(mte_labels, dtype=torch.long)

        return mtr_images, mtr_labels, mte_images, mte_labels

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = idx % self.n_classes
        return idx
