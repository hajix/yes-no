# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np
import scipy as sp
from scipy.signal import argrelextrema
import os
import pygame
import time
from features import mfcc, logfbank
import pickle

# global variables
file_path = './waves_yesno'
Fs = 8000


def segmentation(file_name, win_len=0.025, win_step=0.01, play=False, display=False):

    # read file
    #print(('file name: {0}'.format(file_name)))
    (Fs, data) = wav.read(os.path.join(file_path, file_name))
    data = data[2000:]
    #print(('Fs: {0}\ndata shape: {1}'.format(Fs, data.shape)))

    # calculate frame energies
    win_len = int(win_len * 8000)
    win_step = int(win_step * 8000)
    window = np.hanning(win_len)
    energy = []
    for i in range(0, len(data) - win_len, win_step):
        frame = data[i:i + win_len]
        energy.append([i, 10 * np.log10(np.sum((np.power(frame.astype(float), 2) * window)))])
    energy = np.array(energy, float)

    # energy time smoothing
    energy[:, 1] = sp.convolve(energy[:, 1], np.hanning(21), 'same')
    E = energy[:, 1]

    # find 8 segments
    tr = np.max(E) * 0.87
    bound_E = np.array([x if x > tr else 10 * tr for x in E], float)
    center_E = np.array([x if x > tr else 0.0 for x in E], float)
    centers = argrelextrema(center_E, np.greater)
    bounds = argrelextrema(bound_E, np.less)
    center_indexes = [energy[i, 0] for i in centers][0].tolist()
    bound_indexes = [energy[i, 0] for i in bounds][0].tolist()
    #print(center_indexes)
    #print(bound_indexes)

    # normalize boundaries
    min_seg_len = 2000
    max_seg_len = 2000
    for i in range(len(center_indexes)):
        seg_len = bound_indexes[2 * i + 1] - bound_indexes[2 * i]
        if(seg_len < min_seg_len):
            #print('@smaller', seg_len)
            margin = abs(min_seg_len - seg_len) / 2
            bound_indexes[2 * i] -= margin
            bound_indexes[2 * i + 1] += margin
        if(seg_len > max_seg_len):
            #print('@larger', seg_len)
            margin = abs(max_seg_len - seg_len) / 2
            bound_indexes[2 * i] += margin
            bound_indexes[2 * i + 1] -= margin

    # segmentation
    segments = []
    for start, end in zip(bound_indexes[0::2], bound_indexes[1::2]):
        segments.append(data[start:end])
    segments = np.array(segments)

    # play sound
    if(play):
        print((file_name[:-4].split('_')))
        time.sleep(0.1)
        pygame.mixer.init(Fs, channels=1)
        for i in segments:
            sound_obj = pygame.sndarray.make_sound(i)
            sound_obj.play()
            while pygame.mixer.get_busy():
                time.sleep(0.01)

    # plots
    if(display):
        plt.figure(file_name)
        plt.plot(data)
        #plt.plot(energy[:, 0], bound_E)
        plt.plot(energy[:, 0], center_E)
        plt.show()

    return segments


def make_dataset():
    files = sorted(os.listdir(file_path))
    data_mfcc, data_lmfb, target = [], [], []
    for file_name in files:
        target += [int(x) for x in file_name[:-4].split('_')]
        segments = segmentation(file_name, play=False, display=False)
        for segment in segments:
            data_mfcc.append(mfcc(segment, samplerate=Fs))
            data_lmfb.append(logfbank(segment, samplerate=Fs))
    f = open('dataset.pkl', 'wr+')
    pickle.dump({'data_mfcc': np.array(data_mfcc),
                'data_lmfb': np.array(data_lmfb),
                'target': np.array(target)}, f)
    f.close()
    return {'data_mfcc': np.array(data_mfcc),
            'data_lmfb': np.array(data_lmfb),
            'target': np.array(target)}

if(__name__ == '__main__'):
    files = sorted(os.listdir(file_path))
    for file_name in files:
        parts = file_name[:-4].split('_')
        if(len(parts) != len(segmentation(file_name, play=False))):
            print('error @segmentation')
    print('------------ds----------')
    ds = make_dataset()
    print((max([x.shape[0] for x in ds['data_lmfb']])))
    print((min([x.shape[0] for x in ds['data_lmfb']])))
    print((ds['target']))
