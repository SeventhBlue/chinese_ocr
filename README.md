## 简介
基于Tensorflow和Keras实现端到端的不定长中文字符检测和识别

* 文本检测：CTPN
* 文本识别：DenseNet + CTC

## 环境部署
``` Bash
sh setup.sh
```
* 注：CPU环境执行前需注释掉for gpu部分，并解开for cpu部分的注释

## Demo
将测试图片放入test_images目录，检测结果会保存到test_result中

``` Bash
python demo.py
```

## 模型训练

### CTPN训练
详见ctpn/README.md

### DenseNet + CTC训练

#### 1. 数据准备

数据集：https://pan.baidu.com/s/1QkI7kjah8SPHwOQ40rS1Pw (密码：lu7m)
* 共约364万张图片，按照99:1划分成训练集和验证集
* 数据利用中文语料库（新闻 + 文言文），通过字体、大小、灰度、模糊、透视、拉伸等变化随机生成
* 包含汉字、英文字母、数字和标点共5990个字符
* 每个样本固定10个字符，字符随机截取自语料库中的句子
* 图片分辨率统一为280x32

#### 2. 数据制作

工具：https://github.com/SeventhBlue/textGenerationTool

#### 2. 训练

densenet+ctc训练：https://github.com/SeventhBlue/denseNetTrain

## 参考

[1] https://github.com/YCG09/chinese_ocr
