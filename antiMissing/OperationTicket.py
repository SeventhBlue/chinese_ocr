# -*- coding: utf-8 -*-#
# Author:       weiz
# Date:         2019/11/26 10:53
# Name:         OperationTicket
# Description:  操作票类

import cv2
import numpy as np
import math
import sys

class OperationTicket(object):
    def line_detect(self, image, model=None):
        """
        检测图片中的线
        :param image:
        :param model:
        :return:
        """
        if model != "v":
            model = "horizontal"
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)  # apertureSize是sobel算子大小，只能为1,3,5,7
        # 函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
        longest_l = [0, 0, 0, 0]  # 存储水平最长的线
        # 筛选水平或者竖直方向最长的线
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if model == "horizontal":
                if abs(x2 - x1) > abs(longest_l[2] - longest_l[0]):
                    longest_l = [x1, y1, x2, y2]
            else:
                if abs(y2 - y1) > abs(longest_l[3] - longest_l[1]):
                    longest_l = [x1, y1, x2, y2]
            # cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # cv2.line(image, (longest_l[0], longest_l[1]), (longest_l[2], longest_l[3]), (0, 255, 0), 3)
        # cv2.imshow("line_detect_result", image)
        return longest_l

    def rotate_image(self, img, line=None, degree=None):
        """
        根据line旋转图片
        :param img:
        :param line:
        :param angle:
        :return:
        """
        k = (line[3] - line[1]) / (line[2] - line[0])
        degree = math.atan(k) * 180 / math.pi

        height, width = img.shape[:2]
        # 旋转后图片的宽高
        heightNew = int(
            width * math.fabs(math.sin(math.radians(degree))) + height * math.fabs(math.cos(math.radians(degree))))
        widthNew = int(
            height * math.fabs(math.sin(math.radians(degree))) + width * math.fabs(math.cos(math.radians(degree))))
        # 旋转矩阵
        matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)

        # 加入平移操作,避免将图片旋转到图片区域外
        matRotation[0, 2] += (widthNew - width) // 2
        matRotation[1, 2] += (heightNew - height) // 2

        rotated_img = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(0, 0, 0))

        return rotated_img, matRotation

    def segment_image(self, img, hori_line, vert_line, mat_rot):

        # 得到变换后的像素位置
        hori_point1 = np.dot(mat_rot, np.array([[hori_line[0]], [hori_line[1]], [1]]))
        hori_point2 = np.dot(mat_rot, np.array([[hori_line[2]], [hori_line[3]], [1]]))
        vert_point1 = np.dot(mat_rot, np.array([[vert_line[0]], [vert_line[1]], [1]]))
        vert_point2 = np.dot(mat_rot, np.array([[vert_line[2]], [vert_line[3]], [1]]))
        # cv2.circle(img, (hori_point1[0], hori_point1[1]), 2, (0, 0, 255), 4)
        # cv2.circle(img, (hori_point2[0], hori_point2[1]), 2, (0, 0, 255), 4)
        # cv2.circle(img, (vert_point1[0], vert_point1[1]), 2, (0, 0, 255), 4)
        # cv2.circle(img, (vert_point2[0], vert_point2[1]), 2, (0, 0, 255), 4)

        # 通过排序得到分割图像的左上坐标和右下坐标
        x_loc = [int(hori_point1[0]), int(hori_point2[0]), int(vert_point1[0]), int(vert_point2[0])]
        y_loc = [int(hori_point1[1]), int(hori_point2[1]), int(vert_point1[1]), int(vert_point2[1])]
        x_loc.sort()
        y_loc.sort()
        # left_top_point = (x_loc[0], y_loc[0])
        # right_buttom_point = (x_loc[-1], y_loc[-1])
        # cv2.circle(img, left_top_point, 2, (0, 255, 0), 4)
        # cv2.circle(img, right_buttom_point, 2, (255, 0, 0), 4)
        seg = img[y_loc[0]:y_loc[-1], (x_loc[0] + 4):x_loc[-1]]  # 加4个像素是为了去掉线的宽度

        return seg

    def save_image(self, img, name=None):
        """
        保存图片
        :param img:
        :param name:
        :return:
        """
        if name == None:
            global saveImagePath
            name = saveImagePath
        cv2.imwrite(name, img)

    def BinarizedImage(self, img, thr=None, isReversal=None):
        """
        预处理图片
        :param img: 待预处理的图片
        :param thr: 二值化阈值
        :param isReversal: 是否翻转
        :return:
        """
        if isReversal == None:
            isReversal = True
        if thr == None:
            thr = 110
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thr_img = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
        if isReversal:
            rever_gray = 255 - thr_img
            return rever_gray
        else:
            return thr_img

    def recoTableH(self, img, length=None, color_img=None):
        """
        检测白底黑表格的水平线
        :param img:
        :param length:
        :param color_img: 测试用的参数
        :return:
        """
        if len(img.shape) == 3:
            print("Only accept single channel images！")
            sys.exit()
        if length == None:
            length = 5
        h, w = img.shape
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
        erosion = cv2.erode(img, kernel_h, iterations=1)
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        dilation = cv2.dilate(erosion, kernel_rect, iterations=1)
        # cv2.imshow("dilation", dilation)

        edges = cv2.Canny(dilation, 50, 150, apertureSize=3)
        # cv2.imshow("edges", edges)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=30)
        if np.all(lines == None):
            print("Horizontal line detection is zero!")
            return []

        # 表格的左上和右下坐标
        top = 10000
        left = 10000
        right = 0
        buttom = 0
        h_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if (x2 - x1) != 0:
                k = (y2 - y1) / (x2 - x1)
                if -0.01 < k < 0.01:  # 过滤非水平线
                    if (abs(x2 - x1) / w) < 0.95:  # 过滤长度大于表格的线
                        h_lines.append([x1, y1, x2, y2])
                        top = min([y1, y2, top])
                        left = min([x1, x2, left])
                        right = max([x1, x2, right])
                        buttom = max([y1, y2, buttom])

        table_h_line = []
        table_h_line.append([left, top, right, top])
        h_lines = np.array(h_lines)
        h_lines = h_lines[np.lexsort(h_lines[:, ::-2].T)]  # 按第二列排序
        for [x1, y1, x2, y2] in h_lines:
            table_x1, table_y1, table_x2, table_y2 = table_h_line[-1]
            if y1 - table_y1 > 5:
                table_h_line.append([left, (y1 + 2), right, (y1 + 2)])

        # 测试使用代码
        # if color_img != None:
        #     for [x1, y1, x2, y2] in table_h_line:
        #         cv2.line(color_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        # print(len(table_h_line))
        # cv2.line(color_img, (left, top), (right, top), (0, 255, 0), 2)        # 表格的上水平线
        # cv2.line(color_img, (left, buttom), (right, buttom), (0, 255, 0), 2)  # 表格的下水平线
        # cv2.line(color_img, (left, top), (left, buttom), (0, 255, 0), 2)      # 表格的左竖直线
        # cv2.line(color_img, (right, top), (right, buttom), (0, 255, 0), 2)    # 表格的右竖直线
        # cv2.imshow("erosion", color_img)
        # cv2.waitKey(0)
        return table_h_line

    def recoTableV(self, img, table_h_lines, length=None, color_img=None):
        """
        检测白底黑表格的竖直线
        :param img:
        :param table_h_lines:
        :param length:
        :param color_img: 测试用的参数
        :return:
        """
        if len(img.shape) == 3:
            print("Only accept single channel images！")
            sys.exit()
        if length == None:
            length = 5
        h, w = img.shape
        # cv2.imshow("img", img)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
        erosion = cv2.erode(img, kernel_h, iterations=1)
        # cv2.imshow("erosion", erosion)
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        dilation = cv2.dilate(erosion, kernel_rect, iterations=1)
        # cv2.imshow("dilation", dilation)

        edges = cv2.Canny(dilation, 50, 150, apertureSize=3)
        # cv2.imshow("edges", edges)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=30, maxLineGap=10)
        if np.all(lines == None):
            print("Vertical line detection is zero!")
            return []

        # 表格的左上和右下坐标
        top = table_h_lines[0][1]
        left = table_h_lines[0][0]
        right = table_h_lines[-1][2]
        buttom = table_h_lines[-1][3]
        # print([left, top, right, buttom])
        v_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 2:  # 过滤非竖直的线
                if (abs(y2 - y1) / h) < 0.95 and (left < x1 < right):  # 过滤长度大于表格的线且不是A4纸的边缘
                    v_lines.append([x1, y1, x2, y2])

        table_v_line = []
        table_v_line.append([left, top, left, buttom])
        v_lines = np.array(v_lines)
        v_lines = v_lines[np.lexsort(v_lines[:, ::-1].T)]  # 按第1列排序
        # print(v_lines)
        for [x1, y1, x2, y2] in v_lines:
            table_x1, table_y1, table_x2, table_y2 = table_v_line[-1]
            if x1 - table_x1 > 5:
                table_v_line.append([x1, y1, x2, y2])

        table_v_line.append([right, top, right, buttom])

        return table_v_line

    def isTick_rp(self, img, relativePos, tmp_img_l, thr_similarity=None):
        """
        使用模板匹配的方法检测relativePos区域是否有勾
        :param img:
        :param relativePos:相较于图片的相对位置0到1之间的数，格式是：[[左上角，右下角]]
        :return:
        """
        if thr_similarity == None:
            thr_similarity = 0.15
        methods = [cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED]  # 3种模板匹配方法
        md = methods[2]

        isTickList = []
        h, w = img.shape[:2]
        indt = 1  # 调试数据
        # 对每个区域进行模板匹配
        for ind in range(len(relativePos)):
            xmin = int(relativePos[ind][0] * w)
            ymin = int(relativePos[ind][1] * h)
            xmax = int(relativePos[ind][2] * w)
            ymax = int(relativePos[ind][3] * h)

            ROI = img[ymin:ymax, xmin:xmax]
            # cv2.imshow("isTick", ROI)
            # cv2.waitKey(1000)

            # 每个区域进行多个模板匹配，找到最大的那个
            similarity = 0
            for tmp_img in tmp_img_l:
                # print(indt)
                # print(ROI.shape[:2])
                # print(tmp_img.shape[:2])
                ROI_h, ROI_w = ROI.shape[:2]
                tmp_img_h, tmp_img_w = tmp_img.shape[:2]
                if ROI_h < tmp_img_h or ROI_w < tmp_img_w:
                    # 之所以需要resize，模板匹配要求是待检测(ROI)的区域宽高大于模板的宽高
                    ROI = cv2.resize(ROI, (int(ROI_w * 1.1), int(ROI_h * 1.1)), cv2.INTER_LINEAR)
                result = cv2.matchTemplate(ROI, tmp_img, md)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                if max_val > similarity:
                    similarity = max_val
                indt += 1
            # print(similarity)
            # 如果相似度大于某个阈值就认为是有勾
            if similarity > thr_similarity:
                isTickList.append(1)
            else:
                isTickList.append(0)

        return isTickList

    def isTick(self, img, rectList, tmp_img_l, thr_similarity=None):
        """
        使用模板匹配的方法检测rectList区域是否有勾
        :param img:
        :param rectList: 矩形区域，格式是：[[左上角，右下角]]
        :return:
        """
        if thr_similarity == None:
            thr_similarity = 0.15
        methods = [cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED]  # 3种模板匹配方法
        md = methods[2]

        isTickList = []
        h, w = img.shape[:2]
        indt = 1  # 调试数据
        # 对每个区域进行模板匹配
        for rect in rectList:
            xmin = rect[0]
            ymin = rect[1]
            xmax = rect[2]
            ymax = rect[3]

            ROI = img[ymin:ymax, xmin:xmax]
            # cv2.imshow("isTick", ROI)
            # cv2.waitKey(1000)

            # 每个区域进行多个模板匹配，找到最大的那个
            similarity = 0
            for tmp_img in tmp_img_l:
                # print(indt)
                # print(ROI.shape[:2])
                # print(tmp_img.shape[:2])
                ROI_h, ROI_w = ROI.shape[:2]
                tmp_img_h, tmp_img_w = tmp_img.shape[:2]
                if ROI_h < tmp_img_h or ROI_w < tmp_img_w:
                    # 之所以需要resize，模板匹配要求是待检测(ROI)的区域宽高大于模板的宽高
                    ROI = cv2.resize(ROI, (int(ROI_w * 1.1), int(ROI_h * 1.1)), cv2.INTER_LINEAR)
                result = cv2.matchTemplate(ROI, tmp_img, md)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                if max_val > similarity:
                    similarity = max_val
                indt += 1
            # print(similarity)
            # 如果相似度大于某个阈值就认为是有勾
            if similarity > thr_similarity:
                isTickList.append(1)
            else:
                isTickList.append(0)

        return isTickList


if __name__ == "__main__":
    ot = OperationTicket()
    # print("OperationTicket:{}".format(dir(ot)))

    ot_image = cv2.imread("./operationTicketImages/table_10.jpg")
    # try:
    #     print(ot_image.shape)
    # except:
    #     print("This is not a picture!")
    #     sys.exit()
    src_h, src_w = ot_image.shape[:2]
    #cv2.imshow("input image", ot_image)

    longest_h_l = ot.line_detect(ot_image)
    longest_v_l = ot.line_detect(ot_image, model='v')

    rotate_img, mat_rot = ot.rotate_image(ot_image, longest_h_l)
    #cv2.imshow("rotate image", rotate_img)

    seg_image = ot.segment_image(rotate_img, longest_h_l, longest_v_l, mat_rot)
    cv2.imshow("segment image", seg_image)

    # ot.save_image(seg_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()