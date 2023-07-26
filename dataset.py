import os
import numpy as np
import torch
import pyworld
import soundfile as sf
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset
import torchaudio
import random
from torch.utils.data import DataLoader
import sys
# 1728-point FFT; hop length 130; 600 frames

#sys.path.append('/MyDrive/VSD')
def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
            _, key, _, _, label = line.strip().split(' ')
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list

    elif (is_eval):
        for line in l_meta:
            key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(' ')
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list

#the pad_cut function need to be modified
def pad_cut(sig, max_len=80000):
    sig_len = sig.size()[1]
    if sig_len == max_len:
        return sig
    elif sig_len < max_len:
        # flip and clip
        flip_sig = torch.cat((sig, torch.zeros(sig.size())), dim=1)
        sig_len = 2 * sig_len
        while sig_len < max_len:
            flip_sig = torch.cat((flip_sig, torch.zeros(flip_sig.size())), dim=1)
            sig_len = sig_len * 2
        return flip_sig[:, :max_len]
    else:
        # start_index = random.randint(0, sig_len - max_len)
        # cutted_sig = sig[:, start_index:start_index + max_len]
        cutted_sig=sig[:,0:max_len]
        return cutted_sig

def pad_max(sig,target_len=96000):
    sig_len=sig.size()[1]
    if(sig_len%3==1):
        max_sig=pad_cut(sig,max_len=sig_len+2)
    elif(sig_len%3==2):
        max_sig=pad_cut(sig,max_len=sig_len+1)
    else:
        max_sig=sig
    align_sig=pad_cut(max_sig,target_len)
    align_sig=align_sig.view(align_sig.size(0),-1,3)
    align_sig,_=align_sig.max(dim=2)
    align_sig=align_sig.view(1,-1)
    return align_sig
#
# def pad(x, max_len=64600):
#     x_len = x.shape[0]
#     if x_len >= max_len:
#         return x[:max_len]
#     # need to pad
#     num_repeats = int(max_len / x_len) + 1
#     padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
#     return padded_x


def normalize(tensor):
    # Subtract the mean, and scale to the interval [-1,1]
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean / tensor_minusmean.abs().max()


class Dataset_ASVspoof2019(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)'''

        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, fs = torchaudio.load(self.base_dir + 'processed_2/flac/' + key + '.flac')
        #X, fs = torchaudio.load(self.base_dir + 'flac/' + key + '.flac')

        y = self.labels[key]
        # X = pad_cut(X, 64000)
        # #X=pad_cut(X,96000)
        #
        # np_x = X.squeeze(0).detach().numpy()
        # np_x = np_x.astype('float64')
        # f0,sp,ap=pyworld.wav2world(np_x,fs)
        # coded_x = pyworld.synthesize(f0, sp, ap, fs)
        # coded_x = torch.from_numpy(coded_x).unsqueeze(0)
        # #coded_x=pad_cut(coded_x,32000)
        # coded_x = pad_cut(coded_x, 64000)
        # #coded_x=pad_cut(coded_x,96000)
        # #coded_x=pad_max(coded_x)
        # sf.write(self.base_dir + 'processed_1/flac/' + key + '.flac', coded_x.squeeze(0), fs)
        # sf.write(self.base_dir + 'processed_2/flac/' + key + '.flac', X.squeeze(0), fs)
        # return X,y


        X_v, fs_v = torchaudio.load(self.base_dir + 'processed_1/flac/' + key + '.flac')
        #X_v=pad_cut(X_v,32000)
        #X = pad_max(X)
        return X, X_v, y



class Dataset_ASVspoof2021_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
           '''

        self.list_IDs = list_IDs
        self.base_dir = base_dir

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, fs = torchaudio.load(self.base_dir + 'processed_2/flac/' + key + '.flac')
        #X, fs = torchaudio.load(self.base_dir + 'flac/' + key + '.flac')
        # X=pad_cut(X,64000)
        # #X=pad_cut(X,96000)
        # np_x = X.squeeze(0).detach().numpy()
        # np_x = np_x.astype('float64')
        # f0, sp, ap = pyworld.wav2world(np_x, fs)
        # coded_x = pyworld.synthesize(f0, sp, ap, fs)
        # coded_x = torch.from_numpy(coded_x).unsqueeze(0)
        # #coded_x = pad_cut(coded_x, 32000)
        # #coded_x = pad_cut(coded_x, 64000)
        # coded_x=pad_cut(coded_x,64000)
        # sf.write(self.base_dir + 'processed_1/flac/' + key + '.flac', coded_x.squeeze(0), fs)
        # sf.write(self.base_dir + 'processed_2/flac/' + key + '.flac', X.squeeze(0), fs)
        # return  X, key

        X_v, fs_v = torchaudio.load(self.base_dir + 'processed_1/flac/' + key + '.flac')
        #X_v=pad_cut(X_v,32000)
        #X = pad_max(X)
        return X, X_v,key



if __name__ == '__main__':
    #train data
    d_label_trn, file_train = genSpoof_list(
        dir_meta='./ASVspoof2019.PA.cm.train.trn.txt',
        is_train=True,
        is_eval=False
    )
    print('no. of training trials', len(file_train), flush=True)

    train_set = Dataset_ASVspoof2019(
        list_IDs=file_train,
        labels=d_label_trn,
        base_dir='./ASVspoof2019_PA_train/'
    )
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    for batch_x, batch_y in train_loader:
        print('Batch size: ',batch_x.size())
        #exit(1)
        # print(batch_x.min())

    # valid data
    d_label_dev, file_dev = genSpoof_list(
        dir_meta='./ASVspoof2019.PA.cm.dev.trl.txt',
        is_train=False,
        is_eval=False
    )
    print('no. of validation trials', len(file_dev), flush=True)

    dev_set = Dataset_ASVspoof2019(
        list_IDs=file_dev,
        labels=d_label_dev,
        base_dir='./ASVspoof2019_PA_dev/'
    )
    dev_loader = DataLoader(dev_set, batch_size=128, shuffle=False)
    del dev_set, d_label_dev
    for batch_x, batch_y in dev_loader:
        print(batch_x.max())
        print(batch_x.min())

    # 2021 eval data
    file_eval = genSpoof_list(
        #dir_meta='./ASVspoof2021.PA.cm.eval.progress.trl_1.txt',
        dir_meta='./ASVspoof2021.PA.cm.eval.progress.trl.txt',
        is_train=False,
        is_eval=True
    )
    print('no. of eval trials', len(file_eval), flush=True)
    eval_set = Dataset_ASVspoof2021_eval(list_IDs=file_eval,
                                         base_dir='./ASVspoof2021_PA_eval_progress/')
    eval_loader = DataLoader(eval_set, batch_size=128, shuffle=False)
    i = 0
    for batch_x, batch_y in eval_loader:
        i += 1
        print(batch_x.size())
        print(batch_y)
        #exit(1)

    # #2019 eval data
    d_label_eval, file_eval = genSpoof_list(
        dir_meta='./ASVspoof2019.PA.cm.eval.trl.txt',
        is_train=True,
        is_eval=False
    )
    print('no. of training trials', len(file_eval), flush=True)

    eval_set = Dataset_ASVspoof2021_eval(
        list_IDs=file_eval,
        base_dir='./ASVspoof2019_PA_eval/'
    )
    eval_loader = DataLoader(eval_set, batch_size=128, shuffle=True)
    for batch_x, batch_y in eval_loader:
        print('Batch size: ',batch_x.size())
        #exit(1)
        # print(batch_x.min())


    # X, fs = torchaudio.load('./ASVspoof2021_PA_eval_progress_1/' + 'flac/' + 'PA_E_2040337' + '.flac')
    #
    # X = pad_cut(X,64000)
    #
    # np_x = X.squeeze(0).detach().numpy()
    # np_x = np_x.astype('float64')
    # f0, sp, ap = pyworld.wav2world(np_x, fs)
    # coded_x = pyworld.synthesize(f0, sp, ap, fs)
    # coded_x = torch.from_numpy(coded_x).unsqueeze(0)
    # coded_x = pad_cut(coded_x, 64000)
    # #coded_x=pad_max(coded_x)
    # sf.write( './ASVspoof2021_PA_eval_progress_1/'+ 'processed_1/flac/' + 'PA_E_2040337' + '.flac', coded_x.squeeze(0), fs)