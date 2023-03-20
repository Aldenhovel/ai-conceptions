# Gradient Clip 梯度裁剪

## 有啥用

>**NB:**
>
>深度学习里面的梯度裁剪（Gradient Clipping）是一种防止梯度爆炸或消失的技术，它可以限制梯度的范数或值在一个合理的范围内，从而保证模型的稳定训练。
>
>梯度裁剪有两种常见的方法：
>
>- 按照梯度的L2范数进行裁剪，即如果梯度的L2范数超过了一个阈值，就将梯度缩放到该阈值以下。这种方法可以保持梯度方向不变，只改变大小。
>- 按照梯度的绝对值进行裁剪，即如果梯度的绝对值超过了一个阈值，就将其截断到该阈值以下。这种方法可能会改变梯度方向和大小。
>
>不同的深度学习框架提供了不同的函数来实现梯度裁剪，例如Tensorflow有tf.clip_by_norm和tf.clip_by_global_norm，Pytorch有torch.nn.utils.clip_grad_norm和torch.nn.utils.clip_grad_value，Keras有optimizers.SGD中的clipnorm和clipvalue参数。

## 怎么用



## 代码

```python
def clip_gradient(optimizer, grad_clip):
    # 这里将 optimizer 中需要更新的梯度限制在 (-grad_clip, +grad_clip) 范围之内
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
```

```python
# 梯度裁剪应该放在 loss.backward 和 optimizer.step 之间
loss = ......
optimizer.zero_grad()
loss.backward()
clip_gradient(optimizer, grad_clip)
optimizer.step()
```



