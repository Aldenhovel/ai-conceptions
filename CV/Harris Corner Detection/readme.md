# Harris Corner Detection

## 简介

角点检测(Corner Detection)也称为特征点检测，是图像处理和计算机视觉中用来获取图像局部特征点的一类方法，广泛应用于运动检测、图像匹配、视频跟踪、三维建模以及目标识别等领域中。

<img src="https://ooo.0o0.ooo/2017/06/28/5953bc1427011.jpg" alt="result.jpg" style="zoom:50%;" />

## 局部特征

  不同于HOG、LBP、Haar等基于区域(Region)的图像局部特征，Harris是基于角点的特征描述子，属于feature detector，主要用于图像特征点的匹配(match)，在SIFT算法中就有用到此类角点特征；而HOG、LBP、Haar等则是通过提取图像的局部纹理特征(feature extraction)，用于目标的检测和识别等领域。无论是HOG、Haar特征还是Harris角点都属于图像的局部特征，满足局部特征的一些特性。主要有以下几点：

- 可重复性(Repeatability)：同一个特征可以出现在不同的图像中，这些图像可以在不同的几何或光学环境下成像。也就是说，同一物体在不同的环境下成像(不同时间、不同角度、不同相机等)，能够检测到同样的特征。
- 独特性(Saliency)：特征在某一特定目标上表现为独特性，能够与场景中其他物体相区分，能够达到后续匹配或识别的目的。
- 局部性(Locality)；特征能够刻画图像的局部特性，而且对环境影响因子(光照、噪声等)鲁棒。
- 紧致性和有效性(Compactness and efficiency)；特征能够有效地表达图像信息，而且在实际应用中运算要尽可能地快。

相比于考虑局部邻域范围的局部特征，全局特征则是从整个图像中抽取特征，较多地运用在图像检索领域，例如图像的颜色直方图。
除了以上几点通用的特性外，对于一些图像匹配、检测识别等任务，可能还需进一步考虑图像的局部不变特征。例如尺度不变性(Scale invariance)和旋转不变性(Rotation invariance)，当图像中的物体或目标发生旋转或者尺度发生变换，依然可以有效地检测或识别。此外，也会考虑局部特征对光照、阴影的不变性。

## Harris角点检测

  特征点在图像中一般有具体的坐标，并具有某些数学特征，如局部最大或最小灰度、以及某些梯度特征等。角点可以简单的认为是两条边的交点，比较严格的定义则是在邻域内具有两个主方向的特征点，也就是说在两个方向上灰度变化剧烈。如下图所示，在各个方向上移动小窗口，如果在所有方向上移动，窗口内灰度都发生变化，则认为是角点；如果任何方向都不变化，则是均匀区域；如果灰度只在一个方向上变化，则可能是图像边缘。

![corner.jpg](https://ooo.0o0.ooo/2017/06/28/5953a445031a0.jpg)

## Harris角点性质

- Harris角点检测对亮度和对比度的变化不敏感。
- Harris角点检测具有旋转不变性，但不具备尺度不变性。如下图所示，在小尺度下的角点被放大后可能会被认为是图像边缘。

## python-opencv 代码

```python
import cv2
import numpy as np

filename = "mscoco/val2014/COCO_val2014_000000006874.jpg"
img = cv2.imread(filename)

# 灰度处理
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grayimg = grayimg.astype("float32")

# Harris Corner Detection 参数 2 可以控制模糊度
# dst.shape == grayimg.shape
H_detector = cv2.cornerHarris(grayimg, 2, 3, 0.01)
dst = cv2.dilate(H_detector, None)

# 将高于阈值的点标记为边角点
img[dst > dst.max() * 0.01] = [255, 0, 0]
cv2.imshow('show', img)
```

## 参考文献

[图像特征之Harris角点检测](https://senitco.github.io/2017/06/18/image-feature-harris/)

