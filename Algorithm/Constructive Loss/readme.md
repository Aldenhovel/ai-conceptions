# Constructive Loss

相比较于 $\text{MSE Loss}$ 和 $\text{MAE Loss}$ 直接将结果与目标数值比较的做法，使用 $\text{Constructive Loss}$ 可以更加关注相对距离的学习，常用于对比学习。

$$
Loss=\dfrac{1}{2N}\sum_{n=1}^Nyd^2+(1-y)\cdot\text{max}(margin-d,0)^2
$$

 其中：
 
$$
\begin{cases}
d=||a_n-b_n||_2(欧氏距离)\\
y为样本标签是否匹配(y\in \{0,1\})\\
margin 为边界常数，超参数
\end{cases}
$$

解释：

当 $N=1$ 时：

$$
Loss=yd^2+(1-y)\text{max}(margin-d,0)^2
$$

对于相似样本（$y=1$）有：

$$
Loss=d^2
$$

即相似样本的损失值等于两个特征值在欧氏空间的距离值。

对于不相似样本（$y=0$）有：

$$
Loss=\text{max}(margin-d,0)^2
$$

此时欧氏距离越小反而损失值越大，$margin$ 是为了给 $Loss$ 定下上界，同时也是为了防止训练时走捷径导致 $a_n = b_n$ （不然这种情况下无论样本如何都必然让 $Loss$ 最小化）。

