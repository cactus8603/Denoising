from pathlib import Path
import os
import glob

import numpy as np
import torch
import torch.nn as nn
import torchaudio

from tqdm import tqdm, tqdm_notebook
from torch.utils.data import Dataset, DataLoader
from matplotlib import colors, pyplot as plt
from pypesq import pesq
        

if __name__ == '__main__':
    
    # np.random.seed(999)
    # torch.manual_seed(999)

    # Set if run on cuda 
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # check gpu or cpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(f'Using {device} device')

    torchaudio.set_audio_backend("soundfile")
    # print("TorchAudio backend used:\t{}".format(torchaudio.get_audio_backend()))

    # set para
    sample_rate = 16000
    n_fft = sample_rate * 64
    hop_length = sample_rate * 16

    # Set file path
    NOISE_PATH_DIR = Path('./data/train')
    CLEAN_PATH_DIR = Path('./data/clean')

    noise_file = list(NOISE_PATH_DIR.rglob('*.flac'))
    clean_file = list(CLEAN_PATH_DIR.rglob('*.flac'))

    # div into train and test data
    train_noise_file = noise_file[:int(len(noise_file)/10*6)]
    train_clean_file = clean_file[:int(len(clean_file)/10*6)]

    test_noise_file = noise_file[int(len(noise_file)/10*6):]
    test_clean_file = clean_file[int(len(clean_file)/10*6):]

    # Set dataset
    train_dataset = Dataset(train_noise_file, train_clean_file, n_fft, hop_length)
    test_dataset = Dataset(test_noise_file, test_clean_file, n_fft, hop_length)

    # Set dataloader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True) 

    

