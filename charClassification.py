# -*- coding: utf-8 -*-#
# Author:      weiz
# Date:        2020/1/6 上午11:32
# Name:        charClassification.py
# Description:

import string
import zhon.hanzi

punc_eng = string.punctuation
punc_chn = zhon.hanzi.punctuation
charSeparation = ":|;：；"   # 分隔符

def deleteSepaChar(chnString):
    """
    去掉字符串的分隔符
    :param chnString:
    :return:
    """
    for separChar in charSeparation:
        chnString = chnString.replace(separChar, '')
    return chnString


def is_separation_char(uchar):
    """
    判断一个字符是否是分离字符
    :param uchar:
    :return:
    """
    for char in charSeparation:
        if char == uchar:
            return True
    return False


def is_contain_sep_char(chnString):
    """
    判断一个字符串是否包含分离字符
    :param chnString:
    :return:
    """
    for char in charSeparation:
        if chnString.find(char) > -1:
            return True
    return False


def is_chinese(uchar):
    """
    判断一个unicode是否是汉字
    :param uchar:
    :return:
    """
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
        return True
    else:
        return False


def is_number(uchar):
    """
    判断一个unicode是否是数字
    :param uchar:
    :return:
    """
    if uchar >= u'\u0030' and uchar<=u'\u0039':
        return True
    else:
        return False


def is_alphabet(uchar):
    """
    判断一个unicode是否是英文字母
    :param uchar:
    :return:
    """
    if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):
        return True
    else:
        return False


def is_punctuation(uchar):
    """
    判断某个字符是否是标点符号
    :param uchar:
    :return:
    """
    if (punc_chn.find(uchar) == -1) and (punc_eng.find(uchar) == -1):
        return False
    return True


def delSpaceOfChn(mString):
    """
    删除中文之间的空格。比如“abc ef 哈哈 嘿嘿”变成“abc ef 哈哈嘿嘿”
    :param mString:
    :return:
    """
    space_list = []
    for i in range(len(mString)):
        if mString[i] == ' ':
            space_list.append(i)
    del_space_list = []
    for ind in space_list:
        if ind >= 1:
            if is_chinese(mString[ind-1]):
                if ((ind+1) < len(mString)) and is_chinese(mString[ind+1]):
                    del_space_list.append(ind)
                elif ((ind+2) < len(mString)) and (mString[ind+1] == ' ') and is_chinese(mString[ind+2]):
                    del_space_list.append(ind)
                    del_space_list.append(ind+1)
                elif ((ind+3) < len(mString)) and (mString[ind+1] == ' ') and (mString[ind+2] == ' ') and \
                    is_chinese(mString[ind+3]):
                    del_space_list.append(ind)
                    del_space_list.append(ind+1)
                    del_space_list.append(ind+2)
    del_space_list = sorted(del_space_list, reverse=True)
    new_str = ""
    for i in range(0, len(mString)):
        if i not in del_space_list:
            new_str = new_str + mString[i]

    return new_str


if __name__ == '__main__':
    uchar = '！'
    print(is_punctuation(uchar))
    print(is_chinese(uchar))
    print(is_number(uchar))
    print(is_alphabet(uchar))
    print(deleteSepaChar(";|二|"))