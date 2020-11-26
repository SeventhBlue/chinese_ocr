# -*- coding: utf-8 -*-#
# Author:       weiz
# Date:         2019/10/22 15:20
# Name:         web_service
# Description:  ocr的web服务程序接口

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import ocr
import time
import cv2
from PIL import Image
import numpy as np
from io import BytesIO
import json
import base64
from flask_cors import CORS
import re
import copy
from tkinter import _flatten

import charClassification

from datetime import timedelta

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)
CORS(app)  # 解决Access-Control-Allow-Origin跨域请求问题


# @app.route('/bolian_upload', methods=['POST', 'GET'])
@app.route('/bolian_upload', methods=['POST', 'GET'])  # 添加路由
def bolian_upload():
    upload_file = request.files['file']
    old_file_name = upload_file.filename
    suffix = str(old_file_name).split('.')[-1]
    retData = {}
    if suffix in ['png', 'jpg', 'JPG', 'PNG', 'bmp']:
        file_path = os.path.join('static/images/', old_file_name)
        upload_file.save(file_path)

        image = np.array(Image.open(file_path).convert('RGB'))
        ocrRet = ocr.bolian_model(image)
        retData["detectSign"] = 1
        retData["value"] = ocrRet
    else:
        retData["detectSign"] = 0
        retData["value"] = []

    print("The bolian_upload function is requested!")
    return json.dumps(retData, ensure_ascii=False)


# @app.route('/upload', methods=['POST', 'GET'])
@app.route('/upload', methods=['POST', 'GET'])  # 添加路由
def upload():
    if request.method == 'POST':
        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        user_input = request.form.get("name")

        # 当前文件所在路径
        basepath = os.path.dirname(__file__)

        # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))
        f.save(upload_path)

        #--------------------------------------------------------------------------
        image = np.array(Image.open(upload_path).convert('RGB'))
        # image = cv2.imread(upload_path)
        t = time.time()
        result, image_framed = ocr.modelPicExpansion_h(image)
        output_file = os.path.join("test_result", upload_path.split('/')[-1])
        Image.fromarray(image_framed).save(output_file)
        Image.fromarray(image_framed).save("static/images/test.jpg")

        
        stringResult = {}
        stringResult[0] = "Mission complete, it took {:.3f}s".format(time.time() - t)
        i = 1
        for key in result:
            stringResult[i] = result[key][1].strip('|')
            i = i + 1

        #os.remove(upload_path)  # 删除已识别的图片
        #--------------------------------------------------------------------------
        image_framed = cv2.cvtColor(np.asarray(image_framed), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(basepath, 'static/images', 'test.jpg'), image_framed)
        return render_template('upload_ok.html', userinput=user_input, val1=time.time(), data_dict=stringResult)

    return render_template('upload.html')



@app.route('/ocr_service', methods=['POST', 'GET'])
def detection():
    base64Image = None

    if request.method == 'POST':
        print('POST ')
        rj = request.get_json()
        base64Image = rj['base64Image']
    elif request.method == 'GET':
        print('GET')
        base64Image = request.args['base64Image']

    # print('base64Image=',base64Image)

    byte_date = base64.b64decode(base64Image)
    try:
        imagede = Image.open(BytesIO(byte_date)).convert('RGB')
    except Exception as e:
        print('Open Error! Try again!')
        raise e

    image = np.array(imagede)
    retData = ocr.web_model(image)

    imagede.close()
    return json.dumps(retData, ensure_ascii=False)


@app.route('/otr', methods=['POST', 'GET'])  # 添加路由
def otr():
    if request.method == 'POST':
        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        user_input = request.form.get("name")

        # 当前文件所在路径
        basepath = os.path.dirname(__file__)

        # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))
        f.save(upload_path)

        # --------------------------------------------------------------------------
        # image = np.array(Image.open(upload_path).convert('RGB'))
        image = cv2.imread(upload_path)
        t = time.time()
        ot_image = antiMissingMain.getOTDetectROI(image)
        result, image_framed = ocr.model(ot_image)
        output_file = os.path.join("test_result", upload_path.split('/')[-1])
        Image.fromarray(image_framed).save(output_file)
        Image.fromarray(image_framed).save("static/images/test.jpg")

        stringResult = {}
        stringResult[0] = "Mission complete, it took {:.3f}s".format(time.time() - t)
        i = 1
        for key in result:
            stringResult[i] = result[key][1]
            i = i + 1

        # os.remove(upload_path)  # 删除已识别的图片
        # --------------------------------------------------------------------------
        image_framed = cv2.cvtColor(np.asarray(image_framed), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(basepath, 'static/images', 'test.jpg'), image_framed)
        return render_template('otr_ok.html', userinput=user_input, val1=time.time(), data_dict=stringResult)

    return render_template('otr.html')


@app.route('/imgTemplate', methods=['POST'])  # 添加路由
def imgTemplate():
    if request.method == 'POST':
        d = request.get_json()
        src_w = d["size"]["width"]  # 原图片实际宽高
        src_h = d["size"]["height"]
        base64Image = d["base64Image"]
        key_val = d["params"]
        detectionModel = d["type"]

        print(d)
        byte_date = base64.b64decode(base64Image)
        try:
            imagede = Image.open(BytesIO(byte_date)).convert('RGB')
        except Exception as e:
            print('Open Error! Try again!')
            raise e
        w, h = imagede.size
        image = np.array(imagede)
        # 把截取的图片放大到和原图一样，其余地方用0补充
        if h < src_h and w < src_w:
            img = np.zeros([src_h, src_w, 3], dtype=image.dtype)
            img[0:h, 0:w, :] = image
        else:
            img = image
        if detectionModel == '0':
            result, image_framed, img_table = ocr.dete2img(img)
            for key in result:
                result[key][0] = result[key][0].tolist()

            image_framed = Image.fromarray(image_framed)
            output_buffer = BytesIO()
            image_framed.save(output_buffer, format='JPEG')
            byte_data = output_buffer.getvalue()
            image_code1 = str(base64.b64encode(byte_data))[2:-1]

            img_table = Image.fromarray(img_table)
            output_buffer = BytesIO()
            img_table.save(output_buffer, format='JPEG')
            byte_data = output_buffer.getvalue()
            image_code2 = str(base64.b64encode(byte_data))[2:-1]

            key_val = keywordMatch(key_val, result)

            retData = json.dumps({"srcImage": image_code1, "dstImage": image_code2,
                                  "dstRet": key_val, "srcRet": result}, ensure_ascii=False)
            return retData
        elif detectionModel == '1':
            result, image_framed = ocr.model_1(img)
            for key in result:
                result[key][0] = result[key][0].tolist()

            image_framed = image_framed[0:h, 0:w, :]
            image_framed = Image.fromarray(image_framed)
            output_buffer = BytesIO()
            image_framed.save(output_buffer, format='JPEG')
            byte_data = output_buffer.getvalue()
            image_code = str(base64.b64encode(byte_data))[2:-1]

            for key in result:
                result[key][1] = charClassification.delSpaceOfChn(result[key][1])
            key_val = matchCharacters(key_val, result)

            retData = json.dumps({"srcImage": image_code, "dstRet": key_val, "srcRet": result}, ensure_ascii=False)
            return retData
        else:
            result, image_framed = ocr.model_1(img)
            for key in result:
                result[key][0] = result[key][0].tolist()

            image_framed = image_framed[0:h, 0:w, :]

            image_framed = Image.fromarray(image_framed)
            output_buffer = BytesIO()
            image_framed.save(output_buffer, format='JPEG')
            byte_data = output_buffer.getvalue()
            image_code = str(base64.b64encode(byte_data))[2:-1]

            for key in result:
                result[key][1] = charClassification.delSpaceOfChn(result[key][1])
            key_val = keywordMatch(key_val, result)

            retData = json.dumps({"srcImage": image_code, "dstRet": key_val, "srcRet": result}, ensure_ascii=False)

            return retData


# 方法一：仅仅是通过分割符号进行数据的提炼
def matchCharacters(webChar, ocrRet):
    """
    对识别出的字符进行分割。比如'额定电流：63A'，是通过中间的冒号进行分割的
    :param webChar:
    :param ocrRet:
    :return:
    """
    webChar_copy = copy.deepcopy(webChar)
    if len(webChar) == 0:
        for i, key in enumerate(ocrRet):
            if not charClassification.is_contain_sep_char(ocrRet[key][1]):
                webChar = getRetChar_1(webChar, ocrRet[key], i)
            else:
                webChar = getRetChar_2(webChar, ocrRet[key])
        return webChar
    else:
        for i, key in enumerate(ocrRet):
            deteChn = getChn(ocrRet[key][1])
            deteChn_set = set(deteChn)
            maxRate = 0
            deteString = ""
            for web_key in webChar_copy:
                key_set = set(web_key)
                intersection = deteChn_set & key_set
                if len(intersection) / len(web_key) > maxRate:
                    maxRate = len(intersection) / len(web_key)
                    deteString = web_key
            if maxRate > 0.3:
                webChar[deteString] = getVal(ocrRet[key])
            elif not charClassification.is_contain_sep_char(ocrRet[key][1]):
                webChar = getRetChar_1(webChar, ocrRet[key], i)
            else:
                webChar = getRetChar_2(webChar, ocrRet[key])
        return webChar



keywordsPath = "./keyword.txt"
# 方法二：先通过关键词匹配，再通过分割符进行数据的提炼
def keywordMatch(webChar, ocrRet):
    """
    关键字匹配进行字符分割
    :param webChar:
    :param keyword:
    :return:
    """
    keywords = getKeyword(keywordsPath)

    # 遍历识别出的字符，寻找关键字以及关键字在字符串的下标
    for i, key in enumerate(ocrRet):
        mString = ocrRet[key][1]
        # 数据格式是：[[num, num, keyword]]，前面一个是匹配的关键字的第一个字在字符串中的下标，后一个是关键字的长度，最后一个是关键字
        chn_frag_match_ret = []

        # 遍历关键字表，和识别的出的字符串进行匹配
        for ii in keywords:
            for keyword in keywords[ii][0:-1]:
                if (mString.find(keyword) >= 0) and (keywords[ii][-1] == 0):
                    keywords[ii][-1] += 1
                    chn_frag_match_ret.append([mString.find(keyword), len(keyword), keyword])
        if chn_frag_match_ret:
            chn_frag_match_ret = sorted(chn_frag_match_ret, key=(lambda x: x[0]))
            for ii in range(len(chn_frag_match_ret)):
                if (ii + 1) == len(chn_frag_match_ret):
                    l = chn_frag_match_ret[ii][0] + chn_frag_match_ret[ii][1]
                    deleSepar = charClassification.deleteSepaChar(mString[l:])
                    tmp_map = {"chn": deleSepar, "loc": [int(ocrRet[key][0][0]), int(ocrRet[key][0][1]),
                                                         int(ocrRet[key][0][6]), int(ocrRet[key][0][7])]}
                    webChar[chn_frag_match_ret[ii][2]] = tmp_map
                else:
                    l = chn_frag_match_ret[ii][0] + chn_frag_match_ret[ii][1]
                    u = chn_frag_match_ret[ii + 1][0]
                    deleSepar = charClassification.deleteSepaChar(mString[l:u])
                    tmp_map = {"chn": deleSepar, "loc": [int(ocrRet[key][0][0]), int(ocrRet[key][0][1]),
                                                         int(ocrRet[key][0][6]), int(ocrRet[key][0][7])]}
                    webChar[chn_frag_match_ret[ii][2]] = tmp_map
        elif not charClassification.is_contain_sep_char(ocrRet[key][1]):  # 判断是否含有分隔符
            webChar = getRetChar_1(webChar, ocrRet[key], i)
        else:
            webChar = getRetChar_2(webChar, ocrRet[key])

    keyword_list_old.clear()
    for item in webChar:
        keyword_list_old.append(item)
    return webChar


keyword_list_old = []
@app.route('/writeKeyword', methods=['POST'])  # 添加路由
def writeKeyword():
    if request.method == 'POST':
        d = request.get_json()
        keyword_list_new = []
        for tmp_map in d:
            keyword_list_new.append(tmp_map['key'])
        keyword_list_new_set = set(keyword_list_new)  # 用户更改的关键词

        keywords_loc = readTXT(keywordsPath)
        keyword_list_loc = []
        for item in keywords_loc:
            keyword_list_loc.append(item.split(' '))
        keyword_list_loc = list(_flatten(keyword_list_loc))
        keyword_list_loc_set = set(keyword_list_loc)  # 本地保存所保存的关键词

        keyword_list_old_set = set(keyword_list_old)  # ocr识别时数据被打包的关键词

        write_keywords = keyword_list_new_set - keyword_list_old_set
        write_keywords = list(write_keywords - keyword_list_loc_set)
        print(write_keywords)
        if write_keywords:
            with open(keywordsPath, 'a') as f:  # 'a'表示append,即在原来文件内容后继续写数据
                for keyword in write_keywords:
                    f.write(keyword + '\n')
            info = json.dumps({"info": "已写入字段"}, ensure_ascii=False)
        else:
            info = json.dumps({"info": "已检测字段"}, ensure_ascii=False)
        return info


companyNames = ["有限公司"]
def getRetChar_1(webChar, chnString, i):
    """
    chnString中不含分离字符的数据封装
    :param webChar:
    :param chnString:
    :return:
    """
    for cn in companyNames:
        m = re.search(cn, chnString[1])
        if m:
            tmp_map = {"chn": chnString[1], "loc": [int(chnString[0][0]), int(chnString[0][1]),
                                                    int(chnString[0][6]), int(chnString[0][7])]}
            webChar["制造商"] = tmp_map
        else:
            tmp_map = {"chn": chnString[1], "loc": [int(chnString[0][0]), int(chnString[0][1]),
                                                    int(chnString[0][6]), int(chnString[0][7])]}
            name_map = "其他" + str(i)
            webChar[name_map] = tmp_map
    return webChar


def getRetChar_2(webChar, chnString):
    """
    chnString中含分离字符的数据封装
    :param webChar:
    :param chnString:
    :return:
    """
    tmpChn_key = ""
    tmpChn_val = ""
    flag = 0
    previous_char = 0  # 上个字符是否是分离字符，0表示不是，1表示是
    for char in chnString[1]:
        if not charClassification.is_separation_char(char):
            if flag == 0:
                tmpChn_key += char
            if flag == 1:
                tmpChn_val += char
            previous_char = 0
        else:
            if not previous_char:
                flag += 1
            previous_char = 1
            if flag == 2:
                flag = 0
                tmp_map = {"chn": tmpChn_val, "loc": [int(chnString[0][0]), int(chnString[0][1]),
                                                     int(chnString[0][6]), int(chnString[0][7])]}
                webChar[tmpChn_key] = tmp_map
                tmpChn_key = ""
                tmpChn_val = ""

    tmp_map = {"chn": tmpChn_val, "loc": [int(chnString[0][0]), int(chnString[0][1]),
                                         int(chnString[0][6]), int(chnString[0][7])]}
    webChar[tmpChn_key] = tmp_map
    return webChar


def getChn(mString):
    """
    从字符串中提取所有中文汉字
    :param mString:
    :return:
    """
    chnString = ""
    for char in mString:
        if charClassification.is_chinese(char):
            chnString += char
    return chnString


def getVal(chnString):
    """
    从deteString中得到value
    :param deteString:
    :return:
    """
    tmp_map = {"chn": "", "loc": ""}
    val = ""
    flag = 0
    for char in chnString[1]:
        if flag and (not charClassification.is_separation_char(char)):
            val += char
        elif flag == 0:
            flag += 1
        else:
            flag = 0
    tmp_map["chn"] = val
    tmp_map["loc"] = [int(chnString[0][0]), int(chnString[0][1]),
                      int(chnString[0][6]), int(chnString[0][7])]
    return tmp_map


def readTXT(path):
    """
    读取txt文件
    :param path:
    :return:
    """
    # 按行读取
    with open(path, "r+", encoding='utf-8') as f:
        wordLib = f.readlines()

    # 去掉换行符
    for index in range(len(wordLib)):
        wordLib[index] = wordLib[index].strip('\n')

    return wordLib

def getKeyword(path):
    """
    得到需要匹配的关键字，数据格式比如{0:['出厂日期', '制造日期', '生产日期', '日期', 0]}。最后一位是该项匹配的次数
    :param path:
    :return:
    """
    keyword = readTXT(path)
    keyword_map = {}
    for i, item in enumerate(keyword):
        keyword_map[i] = item.split(' ')
    for i in keyword_map:
        keyword_map[i].append(0)
    return keyword_map


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)   # 本机的IP192.168.0.99