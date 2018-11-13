# -*- coding:utf-8 -*-
# author:zhangwei

"""
   该脚本是用与获取字典文件，并添加CTC所需的blank标签；
"""

def get_dict_list(dictpath):
    list_dict = []
    with open(dictpath) as fr:
        lines = fr.readlines()
        for line in lines:
            list_dict.append(line.strip())
    list_dict.append('_')
    return list_dict

if __name__ =='__main__':
    dictpath = '/home/zhangwei/PycharmProjects/ASR_MFCC//dict_3781'
    a = get_dict_list(dictpath)
    print(a)