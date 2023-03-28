# Xavier initialization

## 原理推导

我们希望网络在训练时有“合理的初始化权值”，即通过适合的参数初始化方法，让张量在网络中可以达到最佳的非线性映射效果。但是在居多网络的层中，例如 `sigmoid` `relu` `batchnorm` 等层都对输入数据的值比较敏感，过大 / 过小的值都可能让输出落入饱和区间，进而失去梯度，如 `sigmoid` 图像：

![img](https://img-blog.csdnimg.cn/20191015192015699.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3hpZXdlbnJ1aTE5OTY=,size_16,color_FFFFFF,t_70)

假设神经网络：
$$
y(x)=w_1x_1+w_2x_2+...+w_nx_n+b
$$
使用标准正态分布初始化分布 $w \sim N(0,1)$ ，那么所有标准正态分布的和分布会变成 $\sum w \sim N(0,n)$ ，随着同层神经网络的权值数量增加，输出 $y$ 的分布也会因为方差上升而逐渐变得不稳定。

从上面 $\sum w$ 的推导可以看出由于 $y$ 均值始终为 0 ，我们从方差 $var(y)$ 入手寻找控制分布偏移的办法，首先：
$$
\begin{align}
var(y)&=var(w_1x_1+w_2x_2+...+w_nx_n+b) \\
&=\sum^n var(w_ix_i)
\end{align}
$$
其中假设 $x$ 和 $w$ 都是高斯分布，他们之积的方差可以转化为：
$$

\begin{align}
var(w_i\cdot x_i) &= \text{E}(w_i^2x_i^2)-\text E(w_ix_i)^2 \\
&=[var(w_i)+\text E(w_i)^2]\cdot[var(x_i)+ \text E(x_i)^2]-\text E(w_i)^2\text E(x_i)^2 \\


\end{align}
$$
代入 $\text E(w_i)=0$ $\text E(x_i)=0$ 有：
$$
var(w_ix_i) = var(w_i)\cdot var(x_i)
$$
所以：
$$
var(y) = n \cdot var(w_i)var(x_i)
$$
假设 $x \sim N(0,1)$ 且 $var(y)=1$ ：
$$
1=n \cdot var(w_i) \rightarrow var(w_i) = 1/n
$$
总结，在上述的推论中，我们假设了输入 $x\sim N(0,1)$ 并想要得到网络在 $n$ 个权值的情况下输出 $y\sim N(0,1)$ ，所需要的条件是将权值 $w$ 初始化为 $w\sim N(0, 1/n)$ 。实际上在原文中作者将层的输入和输出单元均值作为 $n$ 以保持在前向和反向传播中都能保持输出分布稳定：
$$
var(w_i)=1/N_{avg} \quad\text{where}\quad N_{avg}=\dfrac{N_{in}+N_{out}}{2}
$$

## torch 代码

```python
import torch.nn

w = torch.empty(3, 5)
nn.init.xavier_normal_(w)
```

