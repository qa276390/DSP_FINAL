import librosa
import torch
import numpy as np

def pad(raw):
    ex = np.zeros((22050,))
    ex[:len(raw)] = raw
    return ex

def transform(raw, padding=False, simplenorm=False):
    cut = lambda x: x[-11025:]
    norm =  lambda x: (x.astype(np.float32) / (np.max(x)+1e-6))*0.5
    simpnorm = lambda x: x.astype(np.float32) / np.max(x)
    spct = lambda x: librosa.feature.melspectrogram(x) 
    tri = lambda x: [x, x, x]
    totensor = lambda x: torch.Tensor(x)
    
    if padding:
        x = pad(raw)
    else:
        x = cut(raw)
    if simplenorm:
        x = simpnorm(x)
    else:
        x = norm(x)
    x = spct(x)
    x = tri(x)
    x = totensor(x)
    return x