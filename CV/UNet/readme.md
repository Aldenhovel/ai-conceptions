# UNet

![UNet](https://img-blog.csdnimg.cn/20210316213927771.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTA3NDU2OA==,size_16,color_FFFFFF,t_70)

## 简介

1. UNet采用全卷积神经网络。
2. 左边网络为特征提取网络：使用卷积层和池化层。
3. 右边网络为特征融合网络：使用上采样产生的特征图与左侧特征图进行拼接操作。
4. 最后再经过两次卷积操作，生成特征图，再用两个卷积核大小为 $1\times1$ 的卷积做分类得到最后的两张 heatmap ，例如第一张表示第一类的得分，第二张表示第二类的得分heatmap，然后作为 Softmax 函数的输入，算出概率比较大的 Softmax ，然后再进行 loss 反向传播计算。

## 主要效果

- 提出U型的encoder-decoder结构，以逐层恢复的方式复原图像大小。
- 使用 skip-connection 拼接深浅层特征。 为保证拼接时特征的尺寸一致，对 encoder 的特征进行中心裁剪。
- 区别于 FCN **相加** 的融合操作，UNet采用 **拼接** 的方式融合深浅层特征， UNet 的网络结构也非常简单，清晰明了。
- 由于UNet的卷积没有使用padding，其每次卷积后特征图的尺寸都在变化，并且最终输出的分割预测结果尺寸大小比原始图像也要小。

## 代码(Torch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        x = self.conv1(x)
        print('# output shape:', x.shape)
        x = self.conv2(x)
        print('# output shape:', x.shape)
        return x
        

class Down(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.pool(x)
        print('# output shape:', x.shape)
        return x
    

class Up(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=2, stride=2, bias=True),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x1, x2):
        x = torch.cat([self.up(x1), x2], dim=1)
        print('# output shape:', x.shape)
        return x
    
    
class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.encode1 = Conv(in_channels=in_channels, out_channels=64)
        self.down1 = Down()
        self.encode2 = Conv(in_channels=64, out_channels=128)
        self.down2 = Down()
        self.encode3 = Conv(in_channels=128, out_channels=256)
        self.down3 = Down()
        self.encode4 = Conv(in_channels=256, out_channels=512)
        self.down4 = Down()
        self.encode5 = Conv(in_channels=512, out_channels=1024)
        self.up4 = Up(in_channels=1024)
        self.decode4 = Conv(in_channels=1024, out_channels=512)
        self.up3 = Up(in_channels=512)
        self.decode3 = Conv(in_channels=512, out_channels=256)
        self.up2 = Up(in_channels=256)
        self.decode2 = Conv(in_channels=256, out_channels=128)
        self.up1 = Up(in_channels=128)
        self.decode1 = Conv(in_channels=128, out_channels=64)
        self.seghead = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        
    def forward(self, x):
        x_encode1 = self.encode1(x)
        x_encode2 = self.encode2(self.down1(x_encode1))
        x_encode3 = self.encode3(self.down2(x_encode2))
        x_encode4 = self.encode4(self.down3(x_encode3))
        x_encode5 = self.encode5(self.down4(x_encode4))
        x_decode4 = self.decode4(self.up4(x_encode5, x_encode4[..., 4:-4, 4:-4]))
        x_decode3 = self.decode3(self.up3(x_decode4, x_encode3[..., 16:-16, 16:-16]))
        x_decode2 = self.decode2(self.up2(x_decode3, x_encode2[..., 40:-40, 40:-40]))
        x_decode1 = self.decode1(self.up1(x_decode2, x_encode1[..., 88:-88, 88:-88]))
        x = self.seghead(x_decode1)
        print('# output shape:', x.shape)
        
        
if __name__ == '__main__':
    inputs = torch.randn(4, 3, 572, 572)
    print('# input shape:', inputs.shape)
    net = UNet(in_channels=3, num_classes=2)
    output = net(inputs)


# input shape: torch.Size([4, 3, 572, 572])
# output shape: torch.Size([4, 64, 570, 570])
# output shape: torch.Size([4, 64, 568, 568])
# output shape: torch.Size([4, 64, 284, 284])
# output shape: torch.Size([4, 128, 282, 282])
# output shape: torch.Size([4, 128, 280, 280])
# output shape: torch.Size([4, 128, 140, 140])
# output shape: torch.Size([4, 256, 138, 138])
# output shape: torch.Size([4, 256, 136, 136])
# output shape: torch.Size([4, 256, 68, 68])
# output shape: torch.Size([4, 512, 66, 66])
# output shape: torch.Size([4, 512, 64, 64])
# output shape: torch.Size([4, 512, 32, 32])
# output shape: torch.Size([4, 1024, 30, 30])
# output shape: torch.Size([4, 1024, 28, 28])
# output shape: torch.Size([4, 1024, 56, 56])
# output shape: torch.Size([4, 512, 54, 54])
# output shape: torch.Size([4, 512, 52, 52])
# output shape: torch.Size([4, 512, 104, 104])
# output shape: torch.Size([4, 256, 102, 102])
# output shape: torch.Size([4, 256, 100, 100])
# output shape: torch.Size([4, 256, 200, 200])
# output shape: torch.Size([4, 128, 198, 198])
# output shape: torch.Size([4, 128, 196, 196])
# output shape: torch.Size([4, 128, 392, 392])
# output shape: torch.Size([4, 64, 390, 390])
# output shape: torch.Size([4, 64, 388, 388])
# output shape: torch.Size([4, 2, 388, 388])
```

## 参考文献

[UNet详解](https://blog.csdn.net/weixin_45074568/article/details/114901600)

https://zhuanlan.zhihu.com/p/553156653

