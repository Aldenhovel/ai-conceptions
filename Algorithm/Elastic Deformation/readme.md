# 弹性变形 Elastic Deformation

## 简介

弹性变形论文最早是由Patrice等人在2003年ICDAR上发表：[Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis](https://link.zhihu.com/?target=http%3A//cognitivemedium.com/assets/rmnist/Simard.pdf)

<img src="https://www.kaggleusercontent.com/kf/288029/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..xH73szX6k6QctmEiFJekJg.5pI9qGnb6fhsPSpx8qBIN6ZUN97qXT7uWY6I9il10EGHyZ0DfNo3S-n66IjXHnKq-Ekfz49WHW0IA2Mzy9QDuQEm4CJVD7FSCLDQBhQA838jJFa0uLfS3QmtBqjhoaTsQQDChURs14z3xoI-SIoeX66XnH3r6Yk6jei2kIy9Rzgz8rpm6silcn9tsoWJ_0B2yvV4Izr6N24jpDZLKNlrRYiS_zgsIhcPVxUKL77hjnGiuwA_loC-G3beNnnHUXDR0ax0pkgD0xbPLgQ4Jv2J26IR-CQjWmk5s0mdeKVzDFf78dj5Au9q5VmawC4Y7PMACVBI0gGbiqh9UrKyb4C_yeXHIOlU55L3jLiQu65S_PsojxlO3I3A9U1E66kSWu6jp6eoWRlmS4FUTneRoU6qT9WQd9eoT0YQRp3aA9wbpt-9a-UR22o2A8EeIsFQTQrokO78tPtEwe_ug2qTXKqh2pxxqaqgH4gXevKBI8D2PIb-hZMOztWYWKYEbno4fxGzWqIUepCZHv2xnlfFvAslsBmIdRyLYKrm6qy0Ec9D3MvnobeJttG4Zhx21OPtCLsKf8mH1uyutTrOHEG8yxWq92HG6OxrS_CbsHiALsESWHx0P2SZaBfDCW6K5mu13FzZpmm6ks2HqHAO8ZOhvdVj1Z4uiKat_8lU33ZErXj8R6X6gfRMSGwab4r0-cQZeN3U.xVF0MuMi6pCqzJxHBAj-kg/__results___files/__results___4_1.png" alt="img" style="zoom:50%;" />

在生物医学图像上做数据增强有显著作用，例如 UNet 使用了这种方法来提高实例分割性能。

## 代码

```python
import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


def elastic_transform(image, alpha, sigma,
                      alpha_affine, random_state=None):
    """
    alpha: 控制变形强度的变形因子
    sigma: 变形强度服从高斯分布里面 (0, sigma) 的参数 sigma
    alpha_affine: 
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size,
                        center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine,
                                       size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    imageB = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    imageC = map_coordinates(imageB, indices, order=1, mode='constant').reshape(shape)
    return imageC


if __name__ == '__main__':
    img_path = '/home/cxj/Desktop/img/8_5_5.png'
    imageA = cv2.imread(img_path)
    img_show = imageA.copy()
    imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    # Apply elastic transform on image
    imageC = elastic_transform(imageA, imageA.shape[1] * 2,
                                   imageA.shape[1] * 0.08,
                                   imageA.shape[1] * 0.08)

    cv2.namedWindow("img_a", 0)
    cv2.imshow("img_a", img_show)
    cv2.namedWindow("img_c", 0)
    cv2.imshow("img_c", imageC)
    cv2.waitKey(0)
```



## 参考文献

[数据增强：弹性变形](https://zhuanlan.zhihu.com/p/342274228)

