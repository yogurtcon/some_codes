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
