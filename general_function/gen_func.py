# -*- coding:utf-8 -*-
# author:zhangwei

"""
   该脚本是产生评价语音是被准确率的评价指标，采用的是编辑距离；
"""

import difflib

def get_edit_distance(str1 , str2):
    leven_cost = 0
    s = difflib.SequenceMatcher(None , str1 , str2)
    for tag , i1 , i2 , j1 , j2 in s.get_opcodes():
        if tag == 'replace':
            leven_cost += max(i2 - i1 , j2 - j1)
        elif tag =='insert':
            leven_cost += j2 - j1
        elif tag == 'delete':
            leven_cost += i2 - i1
    return leven_cost

if __name__ == '__main__':
    str1 , str2 = '张威是江南大学学生' , '章威是江南学学生'
    a = get_edit_distance(str1 , str2)
    print(a)