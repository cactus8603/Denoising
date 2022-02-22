import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader

n_fft = 64
hop_length = 16

class audioDataset(Dataset):
    def __init__(self, noise, clean, n_fft=64, hop_length=16):
        super().__init__()
        self.noise = noise
        self.clean = clean
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.len_ = len(self.noise)

        self.max_len = 192000  # 16000 per second * 12
    
    def __len__(self):
        return self.len_ 

    def load_audio(self, file):
        wav, _ = torchaudio.load(file)
        return wav

    def __getitem__(self, idx):
        x_noise = self.load_audio(self.noise[idx])
        x_clean = self.load_audio(self.clean[idx])

        x_noise = self._preprocess(x_noise)
        x_clean = self._preprocess(x_clean)

        x_noise_stft = torch.stft(input=x_noise, n_fft=self.n_fft,
                                    hop_length=self.hop_length,
                                    normalized=True)
        x_clean_stft = torch.stft(input=x_clean, n_fft=self.n_fft,
                                    hop_length=self.hop_length,
                                    normalized=True)

        return x_noise_stft, x_clean_stft

    def _preprocess(self, src):
        # To pre process, set 20s per file
        
        src = src.numpy()
        length = src.size
        empty = []
        empty = np.zeros((1, max(self.max_len-length, self.max_len)), dtype='float32')
        processed = np.concatenate((src, empty), axis=1)
        processed = torch.from_numpy(processed)

        return processed
        

    # def __repr__(self):
        # return "<met %s %s>" % (len(self.noise), len(self.clean))

class cconv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0):
        super().__init__()

        self.in_c = in_c
        self.out_c = out_c
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.real_conv = nn.Conv2d(
            in_channels = self.in_c,
            out_channels = self.out_c,
            kernel_size = self.kernel_size,
            padding = self.padding,
            stride = self.stride 
        )
        self.im_conv = nn.Conv2d(
            in_channels = self.in_c,
            out_channels = self.out_c,
            kernel_size = self.kernel_size,
            padding = self.padding,
            stride = self.stride 
        )

        # Initialization (Glorot initialization)
        nn.init.xavier_uniform_(self.real_conv.weight)
        nn.init.xavier_uniform_(self.im_conv.weight)

    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]

        output_real = self.real_conv(x_real) - self.im_conv(x_im)
        output_im = self.im_conv(x_real) + self.real_conv(x_im)

        output = torch.stack((output_real, output_im), dim=-1)
        return output


class cconvTranspose2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, output_padding=0):
        super().__init__()

        self.in_channels = in_c
        self.out_channels = out_c
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        self.real_conv = nn.ConvTranspose2d(
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            kernel_size = self.kernel_size,
            padding = self.padding,
            stride = self.stride,
            output_padding = self.output_padding
        )
        self.im_conv = nn.ConvTranspose2d(
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            kernel_size = self.kernel_size,
            padding = self.padding,
            stride = self.stride,
            output_padding = self.output_padding
        )

        # Initialization (Glorot initialization)
        nn.init.xavier_uniform_(self.real_conv.weight)
        nn.init.xavier_uniform_(self.im_conv.weight)

    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]

        output_real = self.real_conv(x_real) - self.im_conv(x_im)
        output_im = self.im_conv(x_real) + self.real_conv(x_im)

        output = torch.stack((output_real, output_im), dim=-1)
        return output

class batchNorm2d(nn.Module):
    def __init__(self, feature, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()

        self.num_features = feature
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.norm_real = nn.BatchNorm2d(
            num_features = self.num_features,
            eps = self.eps,
            momentum = self.momentum,
            affine = self.affine,
            track_running_stats = self.track_running_stats
        )

        self.norm_im = nn.BatchNorm2d(
            num_features = self.num_features,
            eps = self.eps,
            momentum = self.momentum,
            affine = self.affine,
            track_running_stats = self.track_running_stats
        )

    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 0]
        
        norm_real = self.norm_real(x_real)
        norm_im = self.norm_im(x_im)

        output = torch.stack((norm_real, norm_im), dim=-1)
        return output
    
class encoder(nn.Module):
    def __init__(self, in_c=1, out_c=45, filter_size=(7,5), stride_size=(2,2), padding=(0,0)):
        super().__init__()
        
        self.in_c = in_c
        self.out_c = out_c
        self.filter_size = filter_size
        self.stride_siez = stride_size
        self.padding = padding

        self.conv = cconv2d(
            in_c = self.in_c,
            out_c = self.out_c,
            kernel_size = self.filter_size,
            stride = stride_size,
            padding = self.padding
        )

        self.batch_norm = batchNorm2d(
            feature = self.out_c
        )

        self.leakyRelu = nn.LeakyReLU()

    def forward(self, x):
        _conv = self.conv(x)
        _norm = self.batch_norm(_conv)
        act = self.leakyRelu(_norm)

        return act

class decoder(nn.Module):
    def __init__(self, in_c=1, out_c=45, filter_size=(7,5), stride_size=(2,2), padding=(0,0), output_padding=(0,0), last_layer=False):
        super().__init__()

        self.filter_size = filter_size
        self.stride_siez = stride_size
        self.in_c = in_c
        self.out_c = out_c
        self.padding = padding
        self.last_layer = last_layer
        self.output_padding = output_padding

        self.convt = cconvTranspose2d(
            in_c = self.in_c,
            out_c = self.out_c,
            kernel_size = self.filter_size,
            stride = stride_size,
            padding = self.padding,
            output_padding = self.output_padding
        )

        self.batch_norm = batchNorm2d(
            feature = self.out_c
        )

        self.leakyRelu = nn.LeakyReLU()
    
    def forward(self, x):
        _conv = self.convt(x)

        if (not self.last_layer):
            norm = self.batch_norm(_conv)
            output = self.leakyRelu(norm)
        else:
            phase = _conv / (torch.abs(_conv) + 1e-8)
            mag = torch.tanh(torch.abs(_conv))
            output = phase * mag

        return output


class DCUnet20(nn.Module):
    """
    Deep Complex U-Net class of the model.
    """
    def __init__(self, n_fft=64, hop_length=16):
        super().__init__()
        
        # for istft
        self.n_fft = n_fft
        self.hop_length = hop_length

        # self.set_size(model_complexity=int(45//1.414), input_channels=1, model_depth=20)
        self.set_size(model_complexity=32, input_channels=1, model_depth=20)
        self.encoders = []
        self.model_length = 10

        for i in range(self.model_length):
            print("input channel: ",self.enc_channels[i])
            print("output channel", self.enc_channels[i + 1])
            print("kernel", self.enc_kernel_sizes[i])
            print("stride", self.enc_strides[i])
            print("padding", self.enc_paddings[i])
            print('\n')
            module = encoder(in_c=self.enc_channels[i], 
                             out_c=self.enc_channels[i + 1],
                             filter_size=self.enc_kernel_sizes[i], 
                             stride_size=self.enc_strides[i], 
                             padding=self.enc_paddings[i])
            self.add_module("encoder{}".format(i), module)
            self.encoders.append(module)

        self.decoders = []

        for i in range(self.model_length):
            if i != self.model_length - 1:
                module = decoder(in_c=self.dec_channels[i] + self.enc_channels[self.model_length - i],
                                 out_c=self.dec_channels[i + 1], 
                                 filter_size=self.dec_kernel_sizes[i], 
                                 stride_size=self.dec_strides[i], 
                                 padding=self.dec_paddings[i],
                                 output_padding=self.dec_output_padding[i])
            else:
                module = decoder(in_c=self.dec_channels[i] + self.enc_channels[self.model_length - i], 
                                 out_c=self.dec_channels[i + 1], 
                                 filter_size=self.dec_kernel_sizes[i], 
                                 stride_size=self.dec_strides[i], 
                                 padding=self.dec_paddings[i],
                                 output_padding=self.dec_output_padding[i], 
                                 last_layer=True)
            self.add_module("decoder{}".format(i), module)
            self.decoders.append(module)
       
        
    def forward(self, x, is_istft=True):
        # print('x : ', x.shape)
        orig_x = x
        xs = []
        for i, encoder in enumerate(self.encoders):
            xs.append(x)
            x = encoder(x)
            # print('Encoder : ', x.shape)
            
        p = x
        for i, decoder in enumerate(self.decoders):
            p = decoder(p)
            if i == self.model_length - 1:
                break
            # print('Decoder : ', p.shape)
            p = torch.cat([p, xs[self.model_length - 1 - i]], dim=1)
        
        # u9 - the mask
        
        mask = p
        
        # print('mask : ', mask.shape)
        
        output = mask * orig_x
        output = torch.squeeze(output, 1)


        if is_istft:
            output = torch.istft(output, n_fft=self.n_fft, hop_length=self.hop_length, normalized=True)
        
        return output

    
    def set_size(self, model_complexity, model_depth=20, input_channels=1):

        if model_depth == 20:
            self.enc_channels = [input_channels,
                                 model_complexity,
                                 model_complexity,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 128]

            self.enc_kernel_sizes = [(7, 1),
                                     (1, 7),
                                     (6, 4),
                                     (7, 5),
                                     (5, 3),  
                                     (5, 3),  
                                     (5, 3),  
                                     (5, 3),  
                                     (5, 3), 
                                     (5, 3)] 

            self.enc_strides = [(1, 1),
                                (1, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1)]

            self.enc_paddings = [(0, 0), # (3, 0)
                                 (0, 0), # (0, 3)
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0)]

            self.dec_channels = [0,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity,
                                 model_complexity,
                                 1]

            self.dec_kernel_sizes = [(6, 3), 
                                     (6, 3),
                                     (6, 3),
                                     (6, 4),
                                     (6, 3),
                                     (6, 4),
                                     (8, 5),
                                     (7, 5),
                                     (1, 7),
                                     (7, 1)]

            self.dec_strides = [(2, 1), #
                                (2, 2), #
                                (2, 1), #
                                (2, 2), #
                                (2, 1), #
                                (2, 2), #
                                (2, 1), #
                                (2, 2), #
                                (1, 1),
                                (1, 1)]

            self.dec_paddings = [(0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 3),
                                 (3, 0)]
            
            self.dec_output_padding = [(0,0),
                                       (0,0),
                                       (0,0),
                                       (0,0),
                                       (0,0),
                                       (0,0),
                                       (0,0),
                                       (0,0),
                                       (0,0),
                                       (0,0)]
        else:
            raise ValueError("Unknown model depth : {}".format(model_depth))
         