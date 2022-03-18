from pickle import TRUE
import torch
import gc
from pathlib import Path

from model import *
from function import *
import soundfile as sf

if __name__ == '__main__':
    
    # check gpu or cpu
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(f'Using {device} device')

    n_fft = 64
    hop_length = 16

    # Set file path
    NOISE_PATH_DIR = Path('D:\\others\\桌面\\project\\Denoising\\data\\train')
    CLEAN_PATH_DIR = Path('D:\\others\\桌面\\project\\Denoising\\data\\clean')

    noise_file = list(NOISE_PATH_DIR.rglob('*.flac'))
    clean_file = list(CLEAN_PATH_DIR.rglob('*.flac'))

    # div into train and test data
    train_noise_file = noise_file[:28000] # :int(len(noise_file)/10*6)
    train_clean_file = clean_file[:28000]

    test_noise_file = noise_file[28000:40000] # int(len(noise_file)/10*6):
    test_clean_file = clean_file[28000:40000]

    # Set dataset
    train_dataset = audioDataset(train_noise_file, train_clean_file, n_fft, hop_length)
    test_dataset = audioDataset(test_noise_file, test_clean_file, n_fft, hop_length)

    # Set dataloader
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True) 

    ### test
    # x_noise_stft = train_dataset
    # print(x_noise_stft)
    # for i, data in enumerate(train_loader):
        # print("i:", i)
        # print(data)
        # break
    
    for_test_noise_file = train_noise_file[:2100]
    for_test_clean_file = train_clean_file[:2100]
    for_test_train_dataset = audioDataset(for_test_noise_file, for_test_clean_file, n_fft, hop_length)
    for_test_train_loader = DataLoader(for_test_train_dataset, batch_size=2, shuffle=False)

    for_test_noise_file = test_noise_file[:2100]
    for_test_clean_file = test_clean_file[:2100]
    for_test_test_dataset = audioDataset(for_test_noise_file, for_test_clean_file, n_fft, hop_length)
    for_test_test_loader = DataLoader(for_test_test_dataset, batch_size=4, shuffle=False)
    # print(for_test_test_loader)
    # dcunet20 = DCUnet20(n_fft, hop_length).to(DEVICE)
    # result = getMetLoader(for_test_loader, dcunet20, False)
    # print(result)
    
    """
    # # clear cache
    gc.collect()
    torch.cuda.empty_cache()

    
    # model_weights_path = "output/8/dc20_model_1.pth"
    # checkpoint = torch.load(model_weights_path, map_location=torch.device('cuda'))
    dcunet20 = DCUnet20(n_fft, hop_length).to(DEVICE)
    opt = torch.optim.Adam(dcunet20.parameters())
    loss_func = loss_fn

    # dcunet20.load_state_dict(checkpoint)

    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.8)
    # train_losses, test_losses = train(dcunet20, for_test_train_loader, for_test_test_loader, loss_func, opt, scheduler, 3)
    train_losses, test_losses = train(dcunet20, train_loader, test_loader, loss_func, opt, scheduler, 10)
    """


    
    # for generate ans 
    
    model_weights_path = "output/dc20_model_5.pth"

    dcunet20 = DCUnet20(n_fft, hop_length).to(DEVICE)
    optimizer = torch.optim.Adam(dcunet20.parameters())

    checkpoint = torch.load(model_weights_path,
                            map_location=torch.device('cuda')
                       )

    # noise_files has 1000
    generate_noisy_files = sorted(list(Path("data/test").rglob('*.flac')))
    # noisy_files = Path("data/test/mined_00001.flac")
    clean_padding_file = clean_file[:1000]
    generate_dataset = audioDataset(generate_noisy_files, clean_padding_file, n_fft, hop_length)
    generate_ans_loader = DataLoader(generate_dataset, batch_size=1, shuffle=False)
    dcunet20.load_state_dict(checkpoint)

    # print(len(generate_noisy_files))

    iter_ = iter(generate_ans_loader)
    dcunet20.eval()
    
 
    for idx in tqdm(range(1,1001)):
        
        noise, clean = next(iter_)
        with torch.no_grad():
            gen = dcunet20(noise.to(DEVICE), is_istft=True)
        gen_np = gen[0].view(-1).detach().cpu().numpy()
        sf.write("output/generate/temp/vocal_0{index:04}.flac".format(index=int(idx)), gen_np, 16000)
        del gen
        del gen_np
        torch.cuda.empty_cache()
    
    
    
        


    
    # gen = dcunet20(noise.to(DEVICE), is_istft=True)
    # gen_np = gen[0].view(-1).detach().cpu().numpy()
    # sf.write("generate/vocal_0{idx:04}.flac".format(idx=int(index)), gen_np, 16000)
    # noise_addition_utils.save_audio_file(np_array=x_est_np,file_path=Path("Samples/denoised.wav"), sample_rate=SAMPLE_RATE, bit_precision=16)
