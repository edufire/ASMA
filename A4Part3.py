import os
import sys
import numpy as np
from scipy.signal import get_window
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))
import stft
import utilFunctions as UF

eps = np.finfo(float).eps

"""
A4-Part-3: Computing band-wise energy envelopes of a signal

Write a function that computes band-wise energy envelopes of a given audio signal by using the STFT.
Consider two frequency bands for this question, low and high. The low frequency band is the set of 
all the frequencies between 0 and 3000 Hz and the high frequency band is the set of all the 
frequencies between 3000 and 10000 Hz (excluding the boundary frequencies in both the cases). 
At a given frame, the value of the energy envelope of a band can be computed as the sum of squared 
values of all the frequency coefficients in that band. Compute the energy envelopes in decibels. 

Refer to "A4-STFT.pdf" document for further details on computing bandwise energy.

The input arguments to the function are the wav file name including the path (inputFile), window 
type (window), window length (M), FFT size (N) and hop size (H). The function should return a numpy 
array with two columns, where the first column is the energy envelope of the low frequency band and 
the second column is that of the high frequency band.

Use stft.stftAnal() to obtain the STFT magnitude spectrum for all the audio frames. Then compute two 
energy values for each frequency band specified. While calculating frequency bins for each frequency 
band, consider only the bins that are within the specified frequency range. For example, for the low 
frequency band consider only the bins with frequency > 0 Hz and < 3000 Hz (you can use np.where() to 
find those bin indexes). This way we also remove the DC offset in the signal in energy envelope 
computation. The frequency corresponding to the bin index k can be computed as k*fs/N, where fs is 
the sampling rate of the signal.

To get a better understanding of the energy envelope and its characteristics you can plot the envelopes 
together with the spectrogram of the signal. You can use matplotlib plotting library for this purpose. 
To visualize the spectrogram of a signal, a good option is to use colormesh. You can reuse the code in
sms-tools/lectures/4-STFT/plots-code/spectrogram.py. Either overlay the envelopes on the spectrogram 
or plot them in a different subplot. Make sure you use the same range of the x-axis for both the 
spectrogram and the energy envelopes.

NOTE: Running these test cases might take a few seconds depending on your hardware.

Test case 1: Use piano.wav file with window = 'blackman', M = 513, N = 1024 and H = 128 as input. 
The bin indexes of the low frequency band span from 1 to 69 (69 samples) and of the high frequency 
band span from 70 to 232 (163 samples). To numerically compare your output, use loadTestCases.py
script to obtain the expected output.

Test case 2: Use piano.wav file with window = 'blackman', M = 2047, N = 4096 and H = 128 as input. 
The bin indexes of the low frequency band span from 1 to 278 (278 samples) and of the high frequency 
band span from 279 to 928 (650 samples). To numerically compare your output, use loadTestCases.py
script to obtain the expected output.

Test case 3: Use sax-phrase-short.wav file with window = 'hamming', M = 513, N = 2048 and H = 256 as 
input. The bin indexes of the low frequency band span from 1 to 139 (139 samples) and of the high 
frequency band span from 140 to 464 (325 samples). To numerically compare your output, use 
loadTestCases.py script to obtain the expected output.

In addition to comparing results with the expected output, you can also plot your output for these 
test cases.You can clearly notice the sharp attacks and decay of the piano notes for test case 1 
(See figure in the accompanying pdf). You can compare this with the output from test case 2 that 
uses a larger window. You can infer the influence of window size on sharpness of the note attacks 
and discuss it on the forums.
"""
def computeEngEnv(inputFile, window, M, N, H):
    """
    Inputs:
            inputFile (string): input sound file (monophonic with sampling rate of 44100)
            window (string): analysis window type (choice of rectangular, triangular, hanning, 
                hamming, blackman, blackmanharris)
            M (integer): analysis window size (odd positive integer)
            N (integer): FFT size (power of 2, such that N > M)
            H (integer): hop size for the stft computation
    Output:
            The function should return a numpy array engEnv with shape Kx2, K = Number of frames
            containing energy envelop of the signal in decibles (dB) scale
            engEnv[:,0]: Energy envelope in band 0 < f < 3000 Hz (in dB)
            engEnv[:,1]: Energy envelope in band 3000 < f < 10000 Hz (in dB)
    """
    
    ### your code here
    (fs, x) = UF.wavread(inputFile) # 1 Read Wav File
    w = get_window(window, M)
    mX, pX = stft.stftAnal(x, w, N, H) # 2 Perform STFTAnal
	
    bin_freq = np.arange(N)*fs/float(N) # Bins to Freq
	
    temp1 = np.where(bin_freq>0)[0] # Low Freq 0 < f < 3K 
    temp2 = np.where(bin_freq<3000)[0]
    bins_low = np.intersect1d(temp1, temp2)
    print bins_low[0], bins_low[-1]
	
    temp1 = np.where(bin_freq>3000)[0] # High Freq 3K < f < 10K 
    temp2 = np.where(bin_freq<10000)[0]
    bins_high = np.intersect1d(temp1, temp2)
    print bins_high[0], bins_high[-1]
	
    mXLin = np.power(10,mX/20.0) # 4 Energy Envelopes (FAIL)
    mXlin_dim = mXLin.shape	
    engEnv = np.zeros(mXlin_dim)
    engEnv[:,0] = np.sum(mXLin[:,bins_low]**2, axis = 1)
    engEnv[:,1] = np.sum(mXLin[:,bins_high]**2, axis = 1)
    engEnv = 10*np.log10(engEnv)

    
    maxplotfrequency = 5000.0
    plt.hold(True)
    ax = plt.subplot(2,1,1)
    numFrames = int(mX[:,0].size)
    frameTime = H*np.arange(numFrames)/float(fs)
    binFrequency = fs*np.arange(N*maxplotfrequency/fs)/N
    plt.pcolormesh(frameTime, binFrequency, np.transpose(mX[:,:N*maxplotfrequency/fs+1]))
    plt.xlim([frameTime[0], frameTime[-1]])
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Spectogram')

    ax = plt.subplot(2,1,2)
    plt.plot(frameTime, engEnv[:,0])
    plt.plot(frameTime, engEnv[:,1])
    plt.xlim([frameTime[0], frameTime[-1]])
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Energy (dB)')
    ax.set_title('Energy Envelopes')
    plt.legend(('Low','High'))
    plt.autoscale(tight = True)

    plt.show()
    

    return engEnv
