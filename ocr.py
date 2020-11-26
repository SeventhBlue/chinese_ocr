# -*- coding: utf-8 -*-#
# Author:       weiz
# Date:         2019/11/26 10:22
# Name:         main
# Description:  字符识别的接口
import os
import sys
import cv2
from math import *
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

sys.path.append(os.getcwd() + '/ctpn')
from ctpn.lib.text_connector.text_connect_cfg import Config as TextLineCfg
from ctpn.text_detect import text_detect
from lib.fast_rcnn.config import cfg_from_file
from densenet.model import predict as keras_densenet

import charClassification


def sort_box(box):
    """ 
    对box进行排序
    """
    box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
    return box

def dumpRotateImage(img, degree, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) // 2
    matRotation[1, 2] += (heightNew - height) // 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    ydim, xdim = imgRotation.shape[:2]
    imgOut = imgRotation[max(1, int(pt1[1])) : min(ydim - 1, int(pt3[1])), max(1, int(pt1[0])) : min(xdim - 1, int(pt3[0]))]

    return imgOut

def charRec(img, text_recs, adjust=False):
   """
   加载OCR模型，进行字符识别
   """
   results = {}
   xDim, yDim = img.shape[1], img.shape[0]
    
   for index, rec in enumerate(text_recs):
       xlength = int((rec[6] - rec[0]) * 0.1)
       ylength = int((rec[7] - rec[1]) * 0.2)
       if adjust:
           pt1 = (max(1, rec[0] - xlength), max(1, rec[1] - ylength))
           pt2 = (rec[2], rec[3])
           pt3 = (min(rec[6] + xlength, xDim - 2), min(yDim - 2, rec[7] + ylength))
           pt4 = (rec[4], rec[5])
       else:
           pt1 = (max(1, rec[0]), max(1, rec[1]))
           pt2 = (rec[2], rec[3])
           pt3 = (min(rec[6], xDim - 2), min(yDim - 2, rec[7]))
           pt4 = (rec[4], rec[5])
        
       degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))  # 图像倾斜角度

       partImg = dumpRotateImage(img, degree, pt1, pt2, pt3, pt4)

       if partImg.shape[0] < 1 or partImg.shape[1] < 1 or partImg.shape[0] > partImg.shape[1]:  # 过滤异常图片
           continue

       image = Image.fromarray(partImg).convert('L')
       text = keras_densenet(image)
       
       if len(text) > 0:
           results[index] = [rec]
           results[index].append(text)  # 识别文字
 
   return results


def model(img, adjust=False):
    """
    @img: 图片
    @adjust: 是否调整文字识别结果
    """
    cfg_from_file('./ctpn/ctpn/text.yml')
    text_recs, img_framed, img = text_detect(img)
    text_recs = sort_box(text_recs)
    result = charRec(img, text_recs, adjust)
    return result, img_framed


def model_1(img, adjust=False):
    """
    @img: 图片
    @adjust: 是否调整文字识别结果
    """
    cfg_from_file('./ctpn/ctpn/text.yml')
    text_recs, img_framed, img_expan = text_detect(img)
    text_recs = sort_box(text_recs)
    result = charRec(img_expan, text_recs, adjust)

    f = getScale(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    print(f)
    for key in result:
        for i in range(len(result[key][0])):
            result[key][0][i] = int(result[key][0][i] * 1.0 / f)

    return result, img_framed


def web_model(img, adjust=False):
    """
    @img: 图片
    @adjust: 是否调整文字识别结果
    """
    cfg_from_file('./ctpn/ctpn/text.yml')
    # text_recs数据类型是：[[0,1,2,3,4,5,6,7]] (0,1):左上  (2,3):右上  (4,5):左下  (6,7):右下
    text_recs, img_framed, img = text_detect(img)
    text_recs = sort_box(text_recs)
    result = charRec(img, text_recs, adjust)
    retData = []
    i = 0
    for key in result:
        retData.append({'loc': text_recs[0].tolist(), 'chinese': result[key][1]})
        i = i + 1
    return retData


def bolian_model(img, adjust=False):
    """
    @img: 图片
    @adjust: 是否调整文字识别结果
    """
    cfg_from_file('./ctpn/ctpn/text.yml')
    # text_recs数据类型是：[[0,1,2,3,4,5,6,7]] (0,1):左上  (2,3):右上  (4,5):左下  (6,7):右下
    text_recs, img_framed, img = text_detect(img)
    text_recs = sort_box(text_recs)
    result = charRec(img, text_recs, adjust)
    retData = []
    for key in result:
        retData.append(result[key][1])
    return retData


def getScale(im, scale, max_scale=None):
    """
    获得图片的放缩比例
    :param im:
    :param scale:
    :param max_scale:
    :return:
    """
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return f


def getFontSize(h):
    """
    通过矩形的高大概获得字体的字号
    :param h:
    :return:
    """
    fontSize2h = {}
    fontSize2h[10] = [1, 9]
    fontSize2h[14] = [10, 17]
    fontSize2h[16] = [18, 26]
    fontSize2h[17] = [27, 30]
    fontSize2h[20] = [31, 35]
    fontSize2h[22] = [36, 40]
    fontSize2h[24] = [41, 45]
    fontSize2h[26] = [46, 48]
    fontSize2h[30] = [49, 50]
    fontSize2h[31] = [51, 55]

    for key in fontSize2h:
        if fontSize2h[key][0] <= h <= fontSize2h[key][1]:
            return key
    return 32


def fitTableSize_pil(img, fontsize, font_w, font_h, row, longestChn, gap):
    """
    适配表格大小
    :param img:
    :param fontsize:
    :param font_w:
    :param font_h:
    :param row:
    :param longestChn:
    :param gap:
    :return:
    """
    img_h, img_w = img.shape[:2]

    h_rate = (font_h * row) / img_h
    w_rate = font_w / img_w

    font = ImageFont.truetype("simhei.ttf", fontsize, encoding="utf-8")
    if max(h_rate, w_rate) < 0.9 and min(h_rate, w_rate) > 0.7:
        return font, font_w, font_h

    while (max(h_rate, w_rate) <= 0.7) or (max(h_rate, w_rate) >= 0.9):
        if max(h_rate, w_rate) <= 0.7:
            fontsize = fontsize + 1
            font = ImageFont.truetype("simhei.ttf", fontsize, encoding="utf-8")
            font_w, font_h = font.getsize(longestChn)
            font_w = font_w + 2 * gap
            font_h = font_h + 2 * gap
            h_rate = (font_h * row) / img_h
            w_rate = font_w / img_w
        if max(h_rate, w_rate) >= 0.9:
            fontsize = fontsize - 1
            font = ImageFont.truetype("simhei.ttf", fontsize, encoding="utf-8")
            font_w, font_h = font.getsize(longestChn)
            font_w = font_w + 2 * gap
            font_h = font_h + 2 * gap
            h_rate = (font_h * row) / img_h
            w_rate = font_w / img_w

    font = ImageFont.truetype("simhei.ttf", fontsize, encoding="utf-8")
    font_w, font_h = font.getsize(longestChn)
    font_w = font_w + 2 * gap
    font_h = font_h + 2 * gap
    return font, font_w, font_h


def showChn4table_pil(img, chnList, fontsize=None, gap=None, start_x=None, start_y=None):
    """
    以表格的形式在图片的居中(默认是居中)显示中文或者英文字符
    :param img:
    :param chnList:
    :param fontsize:
    :param gap:表格和字体之间的距离
    :param start_x:
    :param start_y:
    :return:
    """
    row = len(chnList)
    if gap == None:
        gap = 2
    if fontsize == None:
        fontsize = 30
    longestChn = ""
    longestChn_count = 0
    # 涉及到中文汉字，一个汉字占两个字节。但是len()函数只计算字符串的个数，而显示涉及到像素
    for chnStr in chnList:
        count = 0
        for chn in chnStr:
            if charClassification.is_chinese(chn):
                count += 1
        count = (len(chnStr) - count) + count * 2
        if count > longestChn_count:
            longestChn = chnStr
            longestChn_count = count
    img_h, img_w = img.shape[:2]

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    # 如果是在Linux系统下运行，可能需要把simhei.ttf字体放到当前运行脚本同路径下
    font = ImageFont.truetype("simhei.ttf", fontsize, encoding="utf-8")
    font_w, font_h = font.getsize(longestChn)
    font_w = font_w + 2 * gap
    font_h = font_h + 2 * gap

    font, font_w, font_h = fitTableSize_pil(img, fontsize, font_w, font_h, row, longestChn, gap)

    # 修改start_x和start_y就可以在图片的任意位置显示
    if start_x == None:
        start_x = (img_w - font_w) // 2
    if start_y == None:
        start_y = (img_h - font_h * row) // 2
    end_x = start_x + font_w
    end_y = start_y + font_h * row

    for i, chnStr in enumerate(chnList):
        line_spacing = i * font_h
        draw.text((start_x+gap, start_y+line_spacing+gap), chnStr, (0, 0, 0), font)
        draw.line((start_x, start_y+line_spacing, end_x, start_y+line_spacing), 'red')

    draw.line((start_x, end_y, end_x, end_y), 'red')
    draw.line((start_x, start_y, start_x, end_y), 'red')
    draw.line((end_x, start_y, end_x, end_y), 'red')

    img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    return img


def modelPicExpansion_h(img, adjust=False):
    """
    识别结果在图片左侧显示
    @img: 图片
    @adjust: 是否调整文字识别结果
    """
    cfg_from_file('./ctpn/ctpn/text.yml')
    text_recs, img_framed, img_scale = text_detect(img)
    text_recs = sort_box(text_recs)
    result = charRec(img_scale, text_recs, adjust)

    h, w = img.shape[:2]
    img_expan = np.zeros([h, w * 2, 3], dtype=img.dtype)
    img_expan[0:h, 0:w, :] = img
    f = getScale(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)

    for i, list in enumerate(text_recs):
        for j, val in enumerate(list):
            text_recs[i][j] = int(val * 1.0 / f)
    for key in result:
        x1 = w + result[key][0][0]
        y1 = result[key][0][1]
        x2 = w + result[key][0][6]
        y2 = result[key][0][7]
        cv2.rectangle(img_expan, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)
        cv2.rectangle(img_expan, (x1 - w, y1), (x2 - w, y2), (255, 0, 0), thickness=2)
        pilimg = Image.fromarray(img_expan)
        draw = ImageDraw.Draw(pilimg)
        fs = getFontSize((y2 - y1))
        font = ImageFont.truetype("simhei.ttf", fs, encoding="utf-8")
        draw.text((x1 + 2, y1 + 4), result[key][1], (0, 0, 0), font=font)
        img_expan = np.array(pilimg)
    return result, img_expan


def modelPicExpansion_v(img, adjust=False):
    """
    识别结果在图片下侧显示
    @img: 图片
    @adjust: 是否调整文字识别结果
    """
    cfg_from_file('./ctpn/ctpn/text.yml')
    text_recs, img_framed, img_scale = text_detect(img)
    text_recs = sort_box(text_recs)
    result = charRec(img_scale, text_recs, adjust)

    h, w = img.shape[:2]
    img_expan = np.zeros([h * 2, w, 3], dtype=img.dtype)
    img_expan[0:h, 0:w, :] = img
    f = getScale(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)

    for i, list in enumerate(text_recs):
        for j, val in enumerate(list):
            text_recs[i][j] = int(val * 1.0 / f)
    for key in result:
        x1 = result[key][0][0]
        y1 = h + result[key][0][1]
        x2 = result[key][0][6]
        y2 = h + result[key][0][7]
        cv2.rectangle(img_expan, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)
        cv2.rectangle(img_expan, (x1, y1 - h), (x2, y2 - h), (255, 0, 0), thickness=2)
        pilimg = Image.fromarray(img_expan)
        draw = ImageDraw.Draw(pilimg)
        fs = getFontSize((y2 - y1))
        font = ImageFont.truetype("simhei.ttf", fs, encoding="utf-8")
        draw.text((x1 + 2, y1 + 4), result[key][1], (0, 0, 0), font=font)
        img_expan = np.array(pilimg)
    return result, img_expan


def dete2img(img, adjust=False):
    """
    识别结果单独显示
    @img: 图片
    @adjust: 是否调整文字识别结果
    """
    cfg_from_file('./ctpn/ctpn/text.yml')
    text_recs, img_framed, img_scale = text_detect(img)
    text_recs = sort_box(text_recs)
    result = charRec(img_scale, text_recs, adjust)

    h, w = img.shape[:2]
    img_table = np.ones([h, w, 3], dtype=img.dtype) * 255

    f = getScale(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    for key in result:
        for i in range(len(result[key][0])):
            result[key][0][i] = int(result[key][0][i] * 1.0 / f)

    for key in result:
        result[key][1] = charClassification.delSpaceOfChn(result[key][1])
    result_copy = sorted(result.items(), key=lambda x: x[1][0][1], reverse=False)
    chnList = []
    for ind in range(len(result_copy)):
        chnList.append(result_copy[ind][1][1])
    img_table = showChn4table_pil(img_table, chnList)

    return result, img_framed, img_table


if __name__ == "__main__":
    image = np.array(Image.open("./test_images/table_1.jpg").convert('RGB'))
    result, image_framed = modelPicExpansion_v(image)
    image_framed = cv2.cvtColor(np.asarray(image_framed), cv2.COLOR_RGB2BGR)
    cv2.imshow("ret", image_framed)
    cv2.waitKey()