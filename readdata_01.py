# -*- coding:utf-8 -*-
# author:zhangwei

"""
   读取数据文件，采用生成器模型，产生批量数据；
"""

import os
import random
import numpy as np
from general_function.file_wav import *
from general_function.file_dict import *

class DataSpeech():
    def __init__(self , path , type):
        self.datapath = path
        self.type = type
        self.slash = '/'
        if self.slash != self.datapath[-1]:
            self.datapath = self.datapath + self.slash
        self.dic_wavlist = {}                            # {'B22_334': 'data_thchs30/train/B22_334.wav', 'A32_9': 'data_thchs30/train/A32_9.wav'}
        self.dic_textlist = {}                           # {'C14_551': ['从', '严肃', '高雅', '的', '文化', '到']}
        self.wordnum = 0
        self.datanum = 0
        self.list_wavnum = []                            # ['A11_0', 'A11_1', 'A11_10', 'A11_100']
        self.list_textnum = []                           # ['A02_000', 'A02_001', 'A02_002', 'A02_003', 'A02_004']
        self.wavs_data = []
        self.list_text = self.get_text_list()
        self.load_data_list()
        pass

    def load_data_list(self):
        if self.type == 'train':
            filename_wavlist_thchs30 = 'datalist' + self.slash + 'train.wav.lst'
            filename_wordlist_thchs30 = 'datalist' + self.slash + 'train.word.txt'
        elif self.type == 'dev':
            filename_wavlist_thchs30 = 'datalist' + self.slash + 'cv.wav.lst'
            filename_wordlist_thchs30 = 'datalist' + self.slash + 'cv.word.txt'
        elif self.type == 'test':
            filename_wavlist_thchs30 = 'datalist' + self.slash + 'test.wav.lst'
            filename_wordlist_thchs30 = 'datalist' + self.slash + 'test.word.txt'
        else:
            pass
        self.dic_wavlist , self.list_wavnum = get_wav_list(self.datapath + filename_wavlist_thchs30)
        self.dic_textlist , self.list_textnum = get_wav_text(self.datapath + filename_wordlist_thchs30)
        self.datanum = self.get_data_num()
        pass

    def get_data_num(self):
        if len(self.dic_wavlist) == len(self.dic_textlist):
            datanum = len(self.dic_wavlist)
        else:
            datanum = -1
        return datanum

    def get_data(self , n_start):
        filename = self.dic_wavlist[self.list_wavnum[n_start]]
        # print(filename)
        wav_signal , fs = read_wav_data(self.datapath + filename)
        list_text = self.dic_textlist[self.list_textnum[n_start]]
        # print(self.list_textnum[n_start], self.list_wavnum[n_start])
        feat_out = []
        for i in list_text:
            if i != ' ':
                n = self.text2num(i)
                feat_out.append(n)
        data_input = get_frequency_feature(wav_signal , fs)
        data_input = data_input.reshape(data_input.shape[0] , data_input.shape[1] , 1)
        data_label = np.array(feat_out)
        return data_input , data_label

    def data_generator(self , batch_size=16 , audio_length=2000):
        labels = []
        for i in range(0 , batch_size):
            labels.append([0.0])
        labels = np.array(labels , dtype=np.float)
        while True:
            X = np.zeros((batch_size , audio_length , 200 , 1) , dtype=np.float)
            y = np.zeros((batch_size , 64) , dtype=np.int16)
            input_length = []
            label_length = []
            ran_num = random.randint(0 , self.datanum - 1)
            # for i in range(2495):
            #     data_input, data_labels = self.get_data(i)
            #     print(i , data_labels)
            for i in range(batch_size):
                data_input , data_labels = self.get_data((ran_num + i) % self.datanum)
                input_length.append(data_input.shape[0] // 8)                              #卷积修改第一处；
                X[i , 0:len(data_input)] = data_input
                y[i , 0:len(data_labels)] = data_labels
                label_length.append([len(data_labels)])
            label_length = np.array(label_length)
            input_length = np.array(input_length).T
            yield [X , y , input_length , label_length] , labels
        pass

    def get_text_list(self):
        list_dict = []
        with open('/home/zhangwei/PycharmProjects/ASR_MFCC/dict_3781' , 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                list_dict.append(line.strip())
        list_dict.append('_')
        return list_dict

    def get_text_num(self):
        return len(self.list_text)

    def text2num(self , text):
        if text != '':
            return self.list_text.index(text)
        else:
            return self.wordnum

if __name__ == '__main__':
    path = '/home/zhangwei/PycharmProjects/ASR_MFCC/'
    ds = DataSpeech(path , type='train')
    for i in ds.data_generator():
        print(i)
