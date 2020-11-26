# -*- coding: utf-8 -*-#
# Author:       weiz
# Date:         2019/11/26 10:22
# Name:         main
# Description:
import os
import cv2
from antiMissing import OperationTicket
import copy
import numpy as np
import sys

operationTicketImagesPath = "./antiMissing/operationTicketImages"              # "./operationTicketImages"
tickTemplatePath = "./antiMissing/tickTemplate"                                # "./tickTemplate"
saveImagePath = "./antiMissing/processedImages/seg_image.jpg"                  # "./processedImages/seg_image.jpg"
top = 6           # 第top水平线开始
bottom = 23       # 第bottom水平线结束


def getFileName(path=None):
    """
    获得某路径下文件名
    :param path:
    :return:
    """
    if path == None:
        path = operationTicketImagesPath
    fileName = os.listdir(path)
    for i, val in enumerate(fileName):
        fileName[i] = path + '/' + val
    # print(fileName)
    return fileName

def showLine(lineList, img):
    """
    显示直线。线的格式是[[x1,y1,x2,y2]]
    :param lineList:
    :param img:
    :return:
    """
    img = copy.deepcopy(img)
    for [x1, y1, x2, y2] in lineList:
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow("show line", img)
    cv2.waitKey(0)

def read_template(t_templatePath=None):
    """
    读取模板图片
    :return:
    """
    if t_templatePath == None:
        t_templatePath = tickTemplatePath
    name_list = os.listdir(t_templatePath)
    tmp_img_l = []
    for name in name_list:
        path = t_templatePath + '/' + name
        img = cv2.imread(path)
        tmp_img_l.append(img)
    return tmp_img_l

def getROI(h_line, v_line, top, bottom, img=None):
    """
    得到有勾的潜在区域。线的格式是[[x1,y1,x2,y2]]
    :param h_line:
    :param v_line:
    :param top:
    :param buttom:
    :param img: 调试参数
    :return:
    """
    def takeSecond(elem):
        return elem[1]
    h_line.sort(key=takeSecond)  # 对水平线的第二个元素进行排序
    def takeFrist(elem):
        return elem[0]
    v_line.sort(key=takeFrist)   # 对竖直线的第一个元素进行排序

    if len(h_line) > bottom:     # 得到打钩的潜在水平区域
        h_tick = h_line[top:bottom]
    else:
        h_tick = h_line[top:]

    if np.all(img) is not None:
        showLine(h_tick, img)

    rightLine_x = h_tick[0][2]
    secodary = 0
    for line in v_line:
        if 40 < (rightLine_x - line[2]) < 60:
            secodary = line[2]

    if np.all(img) is not None:
        showLine([[secodary, h_tick[0][1], secodary, h_tick[-1][3]]], img)

    rectTick = []    # 矩形框，格式是：[[左上角,右下角]]
    for i, line in enumerate(h_tick):
        if (i + 1) < len(h_tick):
            rectTick.append([secodary, line[1], rightLine_x, h_tick[i+1][1]])

    return rectTick


def getLastTickRect(ot_image, isShow=False):
    """
    得到操作票最后一个勾的区域
    :param ot_image:
    :param isShow:
    :return:
    """
    ot = OperationTicket.OperationTicket()
    # 得到水平和竖直方向最长的线
    longest_h_l = ot.line_detect(ot_image)
    longest_v_l = ot.line_detect(ot_image, model='v')

    # 得到水平方向的线与水平线的夹角，并进行旋转校正
    rotate_img, mat_rot = ot.rotate_image(ot_image, longest_h_l)
    # cv2.imshow("rotate image", rotate_img)

    # 分割出待检测的区域
    seg_image = ot.segment_image(rotate_img, longest_h_l, longest_v_l, mat_rot)

    # 检测水平线
    thr_image = ot.BinarizedImage(seg_image, 150, False)
    table_h_lines = ot.recoTableH(thr_image)
    # showLine(table_h_lines, seg_image)

    # 检测竖直线
    thr_image = ot.BinarizedImage(seg_image, 150, False)
    table_v_lines = ot.recoTableV(thr_image, table_h_lines, color_img=seg_image)
    # showLine(table_v_lines, seg_image)

    detectROI = getROI(table_h_lines, table_v_lines, top, bottom)
    for rect in detectROI:
        cv2.rectangle(seg_image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)

    # 读取模板
    tpl_img_list = read_template()
    # 检测待检测区域是否有勾
    isTickList = ot.isTick(seg_image, detectROI, tpl_img_list)


    lastTickRect = []
    for i in range(len(detectROI) - 1, -1, -1):    # 从后往前遍历
        if isTickList[i]:
            if (0 < (i-1)) and (not isTickList[i-1]):  # 考虑折行情况
                lastTickRect.append([0, detectROI[i-1][1], detectROI[i][2], detectROI[i][3]])
            else:
                lastTickRect.append([0, detectROI[i][1], detectROI[i][2], detectROI[i][3]])
            break

    # 显示结果
    if isShow:
        for ind, rect in enumerate(detectROI):
            if isTickList[ind]:
                cv2.rectangle(seg_image, (rect[0], rect[1]), (rect[2], rect[3]), color=(0, 255, 0), thickness=2)
                cv2.putText(seg_image, "Yes", (rect[0], rect[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            else:
                cv2.rectangle(seg_image, (rect[0], rect[1]), (rect[2], rect[3]), color=(0, 0, 255), thickness=2)
                cv2.putText(seg_image, "No", (rect[0], rect[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.imshow("rect", seg_image)
        cv2.waitKey(0)

    if isShow:
        for rect in lastTickRect:
            cv2.rectangle(seg_image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 255), 2)
        cv2.imshow("last tick", seg_image)
        cv2.waitKey(0)

    return lastTickRect


def getMasterCommand(ot_image, isShow=False):
    """
    得到总调命令
    :param ot_image:
    :param isShow:
    :return:
    """
    ot = OperationTicket.OperationTicket()
    # 得到水平和竖直方向最长的线
    longest_h_l = ot.line_detect(ot_image)
    longest_v_l = ot.line_detect(ot_image, model='v')

    # 得到水平方向的线与水平线的夹角，并进行旋转校正
    rotate_img, mat_rot = ot.rotate_image(ot_image, longest_h_l)
    # cv2.imshow("rotate image", rotate_img)

    # 分割出待检测的区域
    seg_image = ot.segment_image(rotate_img, longest_h_l, longest_v_l, mat_rot)

    # 检测水平线
    thr_image = ot.BinarizedImage(seg_image, 150, False)
    table_h_lines = ot.recoTableH(thr_image)
    if isShow:
        showLine(table_h_lines, seg_image)
    lineSpacing = table_h_lines[1][1] - table_h_lines[0][1]
    masterCommandRect = []
    for i in range(len(table_h_lines)):   # 根据总调命令会跨行的特性，得到总调命令的区域
        if table_h_lines[i+1][1] - table_h_lines[i][1] > (2 * lineSpacing - 10):
            masterCommandRect.append([0, table_h_lines[i][1], table_h_lines[i+1][2], table_h_lines[i+1][3]])
            break

    if isShow:
        for rect in masterCommandRect:
            cv2.rectangle(seg_image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 255), 2)
        cv2.imshow("master command rect", seg_image)
        cv2.waitKey(0)

    return masterCommandRect


def getDetectROI(img):
    """
    获得检测的区域，供其他模块调用
    :param img:
    :return:
    """
    try:
        img.shape
    except:
        print("Incorrect input picture!")
        sys.exit()
    ot = OperationTicket.OperationTicket()
    # 得到水平和竖直方向最长的线
    longest_h_l = ot.line_detect(img)
    longest_v_l = ot.line_detect(img, model='v')

    # 得到水平方向的线与水平线的夹角，并进行旋转校正
    rotate_img, mat_rot = ot.rotate_image(img, longest_h_l)
    # cv2.imshow("rotate image", rotate_img)

    # 分割出待检测的区域
    seg_image = ot.segment_image(rotate_img, longest_h_l, longest_v_l, mat_rot)
    # cv2.imshow("seg_image", seg_image)

    # 总调命令区域
    masterCommand = getMasterCommand(img)
    # 最后一个勾的区域
    lastTick = getLastTickRect(img)

    mask = np.zeros(seg_image.shape[0:2], dtype=np.uint8)
    mask[masterCommand[0][1]:masterCommand[0][3], masterCommand[0][0]:masterCommand[0][2]] = 255
    mask[lastTick[0][1]:lastTick[0][3], lastTick[0][0]:lastTick[0][2]] = 255
    seg_image_mask = cv2.bitwise_and(seg_image, seg_image, mask=mask)

    return seg_image_mask


if __name__ == "__main__":
    print("WorkSpace:{}".format(os.getcwd()))
    ot_filepath = getFileName()
    for path in ot_filepath:
        ot_image = cv2.imread(path)
        try:
            ot_image.shape
        except:
            print("This is not a picture:{}".format(path))
            continue
        #cv2.imshow("ot_image", ot_image)

        # ot = OperationTicket.OperationTicket()
        # # 得到水平和竖直方向最长的线
        # longest_h_l = ot.line_detect(ot_image)
        # longest_v_l = ot.line_detect(ot_image, model='v')
        #
        # # 得到水平方向的线与水平线的夹角，并进行旋转校正
        # rotate_img, mat_rot = ot.rotate_image(ot_image, longest_h_l)
        # # cv2.imshow("rotate image", rotate_img)
        #
        # # 分割出待检测的区域
        # seg_image = ot.segment_image(rotate_img, longest_h_l, longest_v_l, mat_rot)
        # #cv2.imshow("seg_image", seg_image)
        #
        # # 总调命令区域
        # masterCommand = getMasterCommand(ot_image)
        # # 最后一个勾的区域
        # lastTick = getLastTickRect(ot_image)
        #
        # mask = np.zeros(seg_image.shape[0:2], dtype=np.uint8)
        # mask[masterCommand[0][1]:masterCommand[0][3], masterCommand[0][0]:masterCommand[0][2]] = 255
        # mask[lastTick[0][1]:lastTick[0][3], lastTick[0][0]:lastTick[0][2]] = 255
        # seg_image_mask = cv2.bitwise_and(seg_image, seg_image, mask=mask)

        seg_image_mask = getDetectROI(ot_image)

        cv2.imshow("seg_image_mask", seg_image_mask)
        cv2.waitKey(0)
