import torch
import gc
from pathlib import Path

from model import *
from function import *

if __name__ == '__main__':
    
    # check gpu or cpu
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(f'Using {device} device')

    n_fft = 64
    hop_length = 16

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
    train_dataset = audioDataset(train_noise_file, train_clean_file, n_fft, hop_length)
    test_dataset = audioDataset(test_noise_file, test_clean_file, n_fft, hop_length)

    # Set dataloader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True) 

    ### test
    x_noise_stft = train_dataset
    print(x_noise_stft)
    for i, data in enumerate(train_loader):
        print("i:", i)
        # print(data)
        break

    dcunet20 = DCUnet20(n_fft, hop_length).to(DEVICE)
    # result = getMetLoader(train_loader, dcunet20, False)
    # print(result)

    """
    # # clear cache
    gc.collect()
    torch.cuda.empty_cache()

    dcunet20 = DCUnet20(n_fft, hop_length).to(DEVICE)
    opt = torch.optim.Adam(dcunet20.parameters())
    loss_func = loss_fn
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.1)

    train_losses, test_losses = train(dcunet20, train_loader, test_loader, loss_func, opt, scheduler, 4)
    """    