import gc
from matplotlib.style import use
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader

from pesq import pesq
from scipy import interpolate

n_fft = 64
hop_length = 16

# check gpu or cpu
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# loss function : SDR
def loss_fn(x_, y_pred, y_true, eps=1e-8):
    
    x_ = torch.squeeze(x_, 1)
    x = torch.istft(x_, n_fft=n_fft, hop_length=hop_length, normalized=True)
    y_true = torch.squeeze(y_true, 1)
    y_true = torch.istft(y_true, n_fft=n_fft, hop_length=hop_length, normalized=True)

    x = x.flatten(1)
    y_pred = y_pred.flatten(1)
    y_true = y_true.flatten(1)

    def sdr_fn(true, pred, eps=1e-8):
        temp = torch.sum(true * pred, dim=1)
        norm = torch.norm(true, p=2, dim=1) * torch.norm(pred, p=2, dim=1)
        return - (temp / (norm + eps))

    # calculate
    temp_true = x - y_true
    temp_pred = x - y_pred

    temp = torch.sum(y_true**2, dim=1) / (torch.sum(y_true**2, dim=1) + torch.sum(temp_true**2, dim=1) + eps)
    SDR_grade = temp * sdr_fn(y_true, y_pred) + (1-temp) * sdr_fn(temp_true, temp_pred)
    
    return torch.mean(SDR_grade)

def getMetLoader(loader, net, use_net=True):

    wonky_samples = []
    net.eval()
    # Original test metrics
    scale_factor = 41828
    # metric_names = ["CSIG","CBAK","COVL","PESQ","SSNR","STOI","SNR "]
    # metric_names = ["PESQ-WB","PESQ-NB","SNR","SSNR","STOI"]
    metric_names = ["PESQ-WB"]
    # overall_metrics = [[] for i in range(5)]
    overall_metrics = [[]]
    for i, data in enumerate(loader):
        # print("Start: ", i)
        # print(data[0])
        if (i+1)%10==0:
            end_str = "\n"
        else:
            end_str = ","
        #print(i,end=end_str)
        if i in wonky_samples:
            print("Something's up with this sample. Passing...")
        else:
            noisy = data[0]
            clean = data[1]
            
            if use_net: # Forward of net returns the istft version
                x_est = net(noisy.to(DEVICE), is_istft=True)
                x_est_np = x_est.view(-1).detach().cpu().numpy()
            else:
                x_est_np = torch.istft(torch.squeeze(noisy, 1), n_fft=n_fft, hop_length=hop_length, normalized=True).view(-1).detach().cpu().numpy()
                # x_clean_np = torch.istft(torch.squeeze(clean, 1), n_fft=n_fft, hop_length=hop_length, normalized=True).view(-1).detach().cpu().numpy()
            
            # print("use net", (use_net))
            x_clean_np = torch.istft(torch.squeeze(clean, 1), n_fft=n_fft, hop_length=hop_length, normalized=True).view(-1).detach().cpu().numpy()
            
            # metrics = AudioMetrics2(x_clean_np, x_est_np, 16000)
            # print(type(ref_wb), type(x_clean_np))
            # ref_wb = resample(x_clean_np, 48000, 16000)
            # deg_wb = resample(x_est_np, 48000, 16000)
            # ref_wb = x_clean_np
            # deg_wb = x_est_np
            # pesq_wb = pesq(16000, ref_wb, deg_wb, 'wb')

            # print(x_clean_np)
            pesq_wb = pesq(16000, x_clean_np, x_est_np, 'wb')
            # ref_nb = resample(x_clean_np, 48000, 8000)
            # deg_nb = resample(x_est_np, 48000, 8000)
            # ref_nb = x_clean_np
            # deg_nb = x_est_np
            # pesq_nb = pesq(8000, ref_nb, deg_nb, 'nb')

            #print(new_scores)
            #print(metrics.PESQ, metrics.STOI)

            overall_metrics[0].append(pesq_wb)
            # overall_metrics[1].append(pesq_nb)
            # overall_metrics[2].append(metrics.SNR)
            # overall_metrics[3].append(metrics.SSNR)
            # overall_metrics[4].append(metrics.STOI)
    # print()
    # print("Sample metrics computed")
    results = {}
    for i in range(1):
        temp = {}
        temp["Mean"] =  np.mean(overall_metrics[i])
        temp["STD"]  =  np.std(overall_metrics[i])
        temp["Min"]  =  min(overall_metrics[i])
        temp["Max"]  =  max(overall_metrics[i])
        results[metric_names[i]] = temp
    """
    print("Averages computed")
    if use_net:
        addon = "(cleaned by model)"
    else:
        addon = "(pre denoising)"
    print("Metrics on test data",addon)
    """
    # for i in range(5):
        # print("{} : {:.3f}+/-{:.3f}".format(metric_names[i], np.mean(overall_metrics[i]), np.std(overall_metrics[i])))
    print("{} : {:.3f}+/-{:.3f}".format(metric_names, np.mean(overall_metrics), np.std(overall_metrics)))
    return results

def train_epoch(net, train_loader, loss_fn, opt):
    net.train()
    train_ep_loss = 0
    counter = 0

    for x_noise, x_clean in train_loader:
        x_noise, x_clean = x_noise.to(DEVICE), x_clean.to(DEVICE)
        
        # zero gradients
        net.zero_grad()

        x_pred = net(x_noise)

        # calculate loss
        loss = loss_fn(x_noise, x_pred, x_clean)
        loss.backward()
        opt.step()

        train_ep_loss += loss.item()
        counter += 1

        if(counter%500==0):
            print("count", counter)
            print("training loss:{}".format(loss.item()))
    
    train_ep_loss /= counter

    # clear cache
    gc.collect()
    torch.cuda.empty_cache()

    return train_ep_loss

def test_epoch(net, test_loader, loss_fn, use_net=True):
    net.eval()
    test_ep_loss = 0
    counter = 0

    for x_noise, x_clean in test_loader:
        x_noise, x_clean = x_noise.to(DEVICE), x_clean.to(DEVICE)
        x_pred = net(x_noise)

        # calculate loss
        loss = loss_fn(x_noise, x_pred, x_clean)

        test_ep_loss += loss.item()
        counter += 1

        if(counter%500==0):
            print("\ncount", counter)
            print("testing loss:{}".format(loss.item()))
    
    test_ep_loss /= counter

    met = getMetLoader(test_loader, net, use_net)

    # clear cache
    gc.collect()
    torch.cuda.empty_cache()

    return test_ep_loss, met

def train(net, train_loader, test_loader, loss_fn, opt, scheduler, epochs):
    train_loss_record = []
    test_loss_record = []
    for epoch in tqdm(range(epochs)):
        
        """
        if (epoch==0):
            # print("Pre-training evaluation")
            # met = getMetLoader(test_loader, net, False)
            
            with open("./results.txt", "w+") as f:
                f.write("INIT: \n")
                f.write(str(met))
                f.write("\n")
        """
        train_loss = train_epoch(net, train_loader, loss_fn, opt)
        test_loss = 0
        scheduler.step()
        print("Saving model....")
        
        
        with torch.no_grad():
            test_loss, met = test_epoch(net, test_loader, loss_fn, use_net=True)

        train_loss_record.append(train_loss)
        test_loss_record.append(test_loss)

        with open("./results.txt", "a") as f:
            f.write("EPOCH: {} \n{} \n".format(epoch, str(met)))
        

        # save model
        torch.save(net.state_dict(), "output/dc20_model_"+str(epoch+1)+'.pth')
        # torch.save(opt.state_dict(), "output/dc20_opt_"+str(epoch+1)+'.pth')
        
        print("Models saved")

        # clear cache
        torch.cuda.empty_cache()
        gc.collect()
    
    return train_loss, test_loss