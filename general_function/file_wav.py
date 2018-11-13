# -*- coding:utf-8 -*-
# author:zhangwei

"""
   该脚本用于提取语音特征，包括MFCC、FBANK以及语谱图特征；
   该脚本是对标签数据进行处理；
"""

from python_speech_features import mfcc, delta, logfbank
import wave
import numpy as np
from scipy.fftpack import fft

def read_wav_data(filename):
    '''
    获取文件数据以及采样频率；
    输入为文件位置，输出为wav文件数学表示和采样频率；
    '''
    wav = wave.open(filename, 'rb')
    num_frames = wav.getnframes()
    num_channels = wav.getnchannels()
    framerate = wav.getframerate()
    str_data = wav.readframes(num_frames)
    wav.close()
    wave_data = np.fromstring(str_data, dtype=np.short)
    wave_data.shape = -1, num_channels
    wave_data = wave_data.T
    return wave_data, framerate

def get_mfcc_feature(wavsignal, fs):
    '''
    输入为wav文件数学表示和采样频率，输出为语音的MFCC特征+一阶差分+二阶差分；
    '''
    feat_mfcc = mfcc(wavsignal, fs)
    print(feat_mfcc)
    feat_mfcc_d = delta(feat_mfcc, 2)
    feat_mfcc_dd = delta(feat_mfcc_d, 2)
    wav_feature = np.column_stack((feat_mfcc, feat_mfcc_d, feat_mfcc_dd))
    return wav_feature

def get_fbank_feature(wavsignal, fs):
    '''
    输入为wav文件数学表示和采样频率，输出为语音的FBANK特征+一阶差分+二阶差分；
    '''
    feat_fbank = logfbank(wavsignal, fs, nfilt=40)
    feat_fbank_d = delta(feat_fbank, 2)
    feat_fbank_dd = delta(feat_fbank_d, 2)
    wav_feature = np.column_stack((feat_fbank, feat_fbank_d, feat_fbank_dd))
    return wav_feature

def get_frequency_feature(wavsignal, fs):
    '''
    输入为wav文件数学表示和采样频率,输出为语谱图特征，特征维度是200；
    '''
    x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))
    time_window = 25
    wav_array = np.array(wavsignal)
    wav_length = wav_array.shape[1]
    first2end = int(len(wavsignal[0]) / fs * 1000 - time_window) // 10
    data_input = np.zeros(shape=[first2end, 200], dtype=np.float)
    for i in range(0, first2end):
        p_start = i * 160
        p_end = p_start + 400
        data_line = wav_array[0, p_start:p_end]
        data_line = data_line * w
        data_line = np.abs(fft(data_line)) / wav_length
        data_input[i] = data_line[0: 200]
    data_input = np.log(data_input)
    return data_input

def get_wav_list(filepath):
    '''
       读取标签文件，并把标签文件与标签位置进行处理，输入为处理好的标签位置，输出为标签位置列表以及标签位置字典；
    '''
    list_wav = []
    dic_filelist = {}
    with open(filepath, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            res = line.strip().split(' ')
            dic_filelist[res[0]] = res[1]
            list_wav.append(res[0])
    return dic_filelist , list_wav

def get_wav_text(filename):
    '''
       输入为文件位置，输出为标签位置以及标签字典（'D31_984': ['早稻', '播种', '和', '育秧', '的', '天气', '条件', '有利', '与否', '与', '这', '一', '期间', '的', '日', '平均', '温度', '阴雨', '日数', '密切', '相关'], 'D12_867':）；
    '''
    dic_wav_list = {}
    list_text = []
    with open(filename ,'r') as fr:
        lines = fr.readlines()
        for line in lines:
            res = line.strip().split()
            list_text.append(res[0])
            dic_wav_list[res[0]] = res[1 :]
    return dic_wav_list , list_text

if __name__ == '__main__':
    # filepath = 'D4_750.wav'
    filepath = '/home/zhangwei/PycharmProjects/ASR_MFCC/datalist/test.word.txt'
    # wavsignal, fs = read_wav_data(filepath)
    # a = get_mfcc_feature(wavsignal , fs)
    # b = get_fbank_feature(wavsignal , fs)
    # get_frequency_feature(wavsignal , fs)
    # get_wav_list(filepath)
    get_wav_text(filepath)