# 卷积层感受野(Receptive Field)

## 1 概念

CNN 中每一层输出特征图上一个像素点对应输入图像上的区域大小。

## 2 重要性

1. 一般感受野越大模型性能越能得到提升，通常 CNN 最后一层感受野需要大于原图尺寸，这样才能充分利用全局语义信息。
2. 目标检测中 Anchor 大小与感受野要对应，否则影响性能。

## 3 公式

$$
RF_{l+1} = (RF_l-1) \times stride + kernel\_size
$$

## 4 实例

假设神经网络：
$$
img[7, 7, 1] \rightarrow \text{Conv1}(3\times3,1) \rightarrow \text{Conv2}(3\times 3, 1)\rightarrow \text{Conv3}(3\times 3,1) \rightarrow logit
$$

1. $\text{Conv1}$ 中， $RF_1=kernel\_size=3$
2. $\text{Conv2}$ 中，$RF_2=(3-1)\times1+3=5$
3. $\text{Conv3}$ 中，$RF_3=(5-1)\times1+3=7$

在 $\text{Conv3}$ 时感受野 $RF_3$ 等于 $7$ ，满足 $RF_{last} \ge H, W$ ，此神经网络可以完成全局语义信息提取。