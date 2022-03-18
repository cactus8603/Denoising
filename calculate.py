from unicodedata import name
from unittest import main
from scipy.io import wavfile
from pesq import pesq
import soundfile as sf 
from matplotlib import pyplot as plt
from tqdm import trange
import time

def test_pesq():
    ref, rate = sf.read("./data/train/mixed_01001.flac")
    sf.write('./data/clean/ref.wav', ref, samplerate=16000)
    deg, rate = sf.read("./data/train/mixed_01001.wav")
    print(pesq(16000, ref, deg, 'wb'))

    rate, ref = wavfile.read("./data/clean/ref.wav")
    rate, deg = wavfile.read("./data/clean/ref.wav")

    # this contact use wide-band mode
    print(deg)
    print(ref)
    print(pesq(16000, ref, deg, 'wb'))

    ### Test official
    """
    rate, ref = wavfile.read("./data/clean/speech.wav")
    rate, deg = wavfile.read("./data/clean/speech_bab_0dB.wav")

    print(deg)
    print(ref)
    print(pesq(rate, ref, deg, 'wb'))

    """


def print_plot_play(x, Fs):
    # print('%s Fs = %d, x.shape = %s, x.dtype = %s' % (text, Fs, x.shape, x.dtype))
    plt.figure(figsize=(8, 2))
    plt.plot(x, color='gray')
    plt.xlim([0, x.shape[0]])
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()
    # ipd.display(ipd.Audio(data=x, rate=Fs))

if __name__ == '__main__':

    score = 0
    index = []
    for i in trange(1,601):
        # test_pesq()
        ref, rate = sf.read("./data/test/mined_0{index:04}.flac".format(index=int(i)))
        # print_plot_play(x=ref, Fs=rate)

        deg, rate = sf.read("./output/generate/6-1-2/vocal_0{index:04}.flac".format(index=int(i)))
        # print_plot_play(x=deg, Fs=rate)
        # print(deg[:len(ref)])
        # deg = deg[len(ref):]
        
        try:
            score += pesq(16000, ref, deg, 'wb')
            # print(pesq(16000, ref, deg, 'wb'))
            # time.sleep(0.1)
        except KeyboardInterrupt:
            raise
        except :
            index.append(i)
            pass
        # sum += float(pesq(16000, ref, deg, 'wb'))
        # ref, rate = sf.read("./data/clean/speech_bab_0dB.wav")
        # print_plot_play(x=ref, Fs=rate)
        # ref, rate = sf.read("./data/clean/speech.wav")
        # print_plot_play(x=ref, Fs=rate)

    print(score)
    print(index)