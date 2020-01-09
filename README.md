## 提取json文件中的值

```
import json
import os

li = []  # 数据集列表


def load_data(filepath):
    # 遍历filepath下所有文件，包括子目录，路径的最后要加斜杆
    files = os.listdir(filepath)
    for fi in files:
        fi_d = os.path.join(filepath, fi+'/')
        if os.path.isdir(fi_d):
            load_data(fi_d)
        else:
            li.append(fi_d[:-1])
    return li


load_data('D:/语料识别/语料库/')
print(len(li))

for i in range(0, len(li)):
    # file_path = li[i]
    # (filepath, tempfilename) = os.path.split(file_path)
    # (filename, extension) = os.path.splitext(tempfilename)

    b = 'D:/语料识别/语料库/' + str(i) + '.txt'
    # b = filepath + '/' + filename + '.txt'

    file1 = open(li[i], 'r', encoding='utf-8')
    file2 = open(b, 'w', encoding='utf-8')
    for line in file1:
        a_line = json.loads(line)
        b_line = a_line['answer'] + '\n'
        file2.write(b_line)

    print(b)
    file1.close()
    file2.close()

```

## 按固定行数拆分文本

```
# 将一个大文本文件进行拆分，每10000行一次拆分


file1 = open('D:/语料识别/语料库/new2016zh/news2016zh_train.json', 'r', encoding='utf-8')
lines = file1.readlines()
try:
    for j in range(0, (len(lines)//10000)+1):
        file2 = open('D:/语料识别/语料库/new2016zh/train_' + str(j) + '.json', 'w', encoding='utf-8')
        print(10000 * (j + 1), '/', len(lines))
        for line in lines[10000*j: 10000*(j+1)]:
            file2.write(line)
        file2.close()
finally:
    file1.close()

```

## 遍历filepath下所有文件，包括子目录，存在于同一文件夹下的文件即拥有同一标签的数据
```markdown
import os
import cv2
import numpy as np
import matplotlib.image as mi


dataset = []  # 数据集列表
labels = []  # 标签列表
label = 0  # 第一个标签


def load_data(filepath):
    files = os.listdir(filepath)
    for fi in files:
        fi_d = os.path.join(filepath, fi+'/')
        if os.path.isdir(fi_d):
            global label
            load_data(fi_d)
            label += 1
        else:
            labels.append(label)
            img = mi.imread(fi_d[:-1])
            img2 = cv2.resize(img, (64, 64))
            dataset.append(img2)

    return np.array(dataset), np.array(labels)
```

## cmd里面直接用 pip install 不行时，用这句
```markdown
python -m pip install
```

## 希望实现 “保存与加载模型” 这一功能

`new_model = keras.models.load_model('my_model.h5')`

报错：
```markdown
ValueError: Unknown initializer: GlorotUniform
```
解决方法是在‘keras’前面加‘tensorflow.’
```markdown
tensorflow.keras.models.load_model('my_model.h5')
```

## 希望实现 “导入图片时将图片尺寸变成28*28” 这一功能
```markdown
img = mpimg.imread('1.png')
img2 = img.resize((28,28))
```
报错: 
```markdown
AttributeError: 'NoneType' object has no attribute 'shape'
```
解决办法是导入cv2，用他的resize方法
```markdown
import cv2

img = mpimg.imread('1.png')
img2 = cv2.resize(img, (28,28))
```

## pip安装时ReadTimeoutError解决办法
```markdown
pip --default-timeout=100 install
```

## pip使用国内源，直接使用 - i 来指定使用哪个 url，如下所示：

阿里云 http://mirrors.aliyun.com/pypi/simple/

中国科技大学 https://pypi.mirrors.ustc.edu.cn/simple/

豆瓣 http://pypi.douban.com/simple/

清华大学 https://pypi.tuna.tsinghua.edu.cn/simple/

中国科学技术大学 http://pypi.mirrors.ustc.edu.cn/simple/

```markdown
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple XXX
```

## python中 **tf.app.flags** 用法
```markdown
import tensorflow as tf
 
第一个是参数名称，第二个参数是默认值，第三个是参数描述
tf.app.flags.DEFINE_string("job_name", "coder", "name of job")
FLAGS = tf.app.flags.FLAGS
 
def main(_):  
    print(FLAGS.job_name)
 
if __name__ == '__main__':
    tf.app.run()  #执行main函数
```

**在控制台打开该python文件**

直接打开:
```markdown
python test.py

得到结果
coder
```

添加后缀：
```markdown
python test.py --job_name=programmer

得到结果
programmer
```

## 使用方法model.predict_classes()对输入的图片进行预测时报错：

ValueError: Error when checking input:expected conv2d_1_input to have 4 dimensions, but got array with shape (X, X, X)

```markdown
img = mpimg.imread('data/mfcc_image_ts/0/0_yicong_20.png')
img2 = np.zeros((1,img.shape[0],img.shape[1],img.shape[2]))
img2[0, :] = img
pre = model.predict_classes(img2)
result = pre[0]
print(result)

原因是img加载进来是(250, 250, 4)，但是方法predict_classes()要求输入的第一个dimension是bachsize

所以需要将数据reshape为(1，X, X, X)

四个参数分别对应图片的个数，图片的通道数，图片的长与宽。具体的参加keras文档
```

## 单用model.evaluate方法不会自动输出数值，需要手动输出他返回的两个数值，如下所示
```markdown
test_scores = model.evaluate(test_image, test_label)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])
```

## 数据的预处理不是必须的

灰度化，归一化，数组降维等等操作都会导致特征丢失

并不是每一次训练都要做这些处理，每做一次处理都可能影响准确率

什么时候，需不需要做这些处理，需要视情况而定

## Sklearn的train_test_split函数

用于将矩阵随机划分为训练子集和测试子集，并返回划分好的训练集测试集样本和训练集测试集标签

```markdown
x_train,x_test, y_train, y_test =train_test_split(x,y,test_size=0.25, random_state=0)
```

参数解释：
```markdown
train_data：被划分的样本特征集

train_target：被划分的样本标签

test_size：如果是浮点数，在0-1之间，表示样本占比；如果是整数的话就是样本的数量

random_state：是随机数的种子。
```

## 将两个数组打乱顺序，但是样本和标签仍然对应
```markdown
from sklearn.utils import shuffle

a = [0, 1, 2]
b = [7, 8, 9]

(a, b) = shuffle(a, b)
```

## 用python代码 将json文件转化为csv文件

path输入目标文件的路径，但是最后的后缀名不要打进去
```markdown
import csv
import json
import sys
import codecs


def trans(path):
    jsonData = codecs.open(path + '.json', 'r', 'utf-8')
    csvfile = open(path + '.csv', 'w', newline='')  # python3下
    # 这里添加quoting=csv.QUOTE_ALL的作用是将所有内容用引号包括起来，其
    # 作用是可以防止在用excel这类软件打开的时候，会自动判断内容里面的逗号
    # 就会将其识别为另外的列
    writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_ALL)
    flag = True
    line_num = 0
    # 取出的每一行数据
    # 这样是为了读取比较大的文件的时候，不会造成内存溢出的问题
    with codecs.open(path + '.json', 'r', 'utf-8') as jsonData:
        for line in jsonData:
            dic = json.loads(line[0:-1])
            # 关键字，但是这个操作只执行一次，因为csv只有表头是需要这个的
            if flag:
                # 获取属性列表
                keys = list(dic.keys())
                print(keys)
                writer.writerow(keys)  # 将属性列表写入csv中，是写入一行，应该会自动换行
                flag = False
            print(list(dic.values()))
            writer.writerow(list(dic.values()))
            line_num += 1
    jsonData.close()
    csvfile.close()
    print('total line is', line_num)


if __name__ == '__main__':
    path = input('enter path:')
    trans(path)
    print('end！')

```

## python列表处理

```markdown
str_a = 'abcde'
print(str_a)  # 输出字符串  abcde
print(str_a[0:-1])  # 输出第一个到倒数第二个的所有字符  abcd
print(str_a[0])  # 输出字符串第一个字符  a
print(str_a[2:5])  # 输出从第三个开始到第五个的字符  cde
print(str_a[2:])  # 输出从第三个开始的后的所有字符  cde
print(str_a * 2)  # 输出字符串两次  abcdeabcde
print(str_a + "TEST")  # 连接字符串  abcdeTEST


list_a = [0, 1, 2, 3, 4, 5]
list_b = [6, 7, 8]
print(list_a)  # 输出完整列表  [0, 1, 2, 3, 4, 5]
print(list_a[0])  # 输出列表第一个元素  0
print(list_a[1:3])  # 从第二个开始输出到第三个元素  [1, 2]
print(list_a[:2])  # 从第一个元素开始输出到第二个元素  [0, 1]
print(list_a[2:])  # 输出从第三个元素开始的所有元素  [2, 3, 4, 5]
print(list_a[-3:-1])  # 从倒数第三个元素开始输出到倒数第二个元素  [3, 4]
print(list_a[-3:])  # 输出从倒数第三个元素开始的所有元素  [3, 4, 5]
print(list_b * 2)  # 输出两次列表  [6, 7, 8, 6, 7, 8]
print(list_a + list_b)  # 连接列表  [0, 1, 2, 3, 4, 5, 6, 7, 8]


dict_a = {'one': 1,
          2: 'two',
          3: 3,
          'four': 'four'}
print(dict_a['one'])  # 输出键为 'one' 的值  1
print(dict_a[2])  # 输出键为 2 的值  two
print(dict_a)  # 输出完整的字典  {'one': 1, 2: 'two', 3: 3, 'four': 'four'}
print(dict_a.keys())  # 输出所有键  dict_keys(['one', 2, 3, 'four'])
print(dict_a.values())  # 输出所有值  dict_values([1, 'two', 3, 'four'])
```
