import numpy as np
import random
import torch
from scipy.signal import hilbert
import torch

def get_mean_rf(sum_rf_list, avg_size, mode = 'continuous') :
    want_signals = torch.stack(list(sum_rf_list))
    mean_signal = torch.mean(want_signals[:avg_size], 0)
    #print(mean_signal.shape)
    
    return mean_signal

def make_hilbert_signal(signal) :
    x = signal.clone()
    for channel in range(x.shape[0]) : 
        start = x[channel][0]
        x[channel] = x[channel].clone() - start
    h2 = hilbert(x)#.imag
    h2 = torch.from_numpy(h2)
    e = torch.abs(h2)
    e = e.float()
    #out_list.append(e)
    #output = torch.stack(out_list, 0)
    #e = (x**2+h2**2) ** 0.5
    
    return e
