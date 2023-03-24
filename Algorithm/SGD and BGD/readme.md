# Stochastic Gradient Descent (SGD) 随机梯度下降

## Introducing SGD

>**AI:**
>
>Stochastic Gradient Descent (SGD) is a popular optimization algorithm used in machine learning for finding the minimum of a cost function. It is a variant of the gradient descent algorithm that iteratively updates the parameters of a model by computing the gradient of the cost function with respect to the parameters using a small subset of the training data at each iteration.
>
>In other words, instead of computing the gradient of the cost function for the entire dataset, SGD randomly selects a small batch of training data at each iteration and computes the gradient of the cost function only on that batch. This results in faster convergence and reduced computation time since the algorithm doesn't need to process the entire dataset in each iteration.
>
>SGD can be used for a wide range of machine learning models, including linear regression, logistic regression, and neural networks. However, it requires careful tuning of the learning rate, which determines the step size in each iteration, to ensure convergence to the optimal solution. Additionally, the random sampling of batches can lead to fluctuations in the cost function, which can be addressed by using a variant of SGD called mini-batch gradient descent that uses a small fixed batch size instead of a single data point.

SGD 实际上就是在一个 epoch 中每个 batch 做一次梯度更新，因为数据集每个 epoch 会被打乱，于是就相当于是随机样本子集对模型进行权值更新。

## Introducing BGD

>**AI:**
>
>Batch Gradient Descent is an optimization algorithm used in machine learning for finding the optimal values of the parameters of a model by iteratively updating the parameters in the direction of the negative gradient of the cost function using the entire dataset at each iteration.
>
>In batch gradient descent, the cost function is computed over the entire dataset, and then the gradient of the cost function is calculated with respect to each parameter. The gradient is then used to update the parameters in the direction that reduces the cost function. The process is repeated for a fixed number of iterations or until convergence is achieved.
>
>The main advantage of batch gradient descent is that it guarantees convergence to the global minimum, provided the cost function is convex. However, it can be computationally expensive and may not be practical for very large datasets that cannot fit into memory. Additionally, batch gradient descent can get stuck in local minima or plateaus, making it less effective for some cost functions.
>
>Batch gradient descent is also known as "vanilla" gradient descent or **"full-batch"** gradient descent to distinguish it from other variants such as stochastic gradient descent and mini-batch gradient descent, which update the parameters using a subset of the data at each iteration.

对比 Batch Gradient Descent，SGD 在每个 batch 计算完成后立即更新网络权值，而 BGD 需要汇总所有 batch 的 loss 信息再进行权值更新，在实际使用上会有很多限制。

## SGD Advantages (vs. BGD)

>**AI:**
>
>SGD is more popular than batch gradient descent because of the following reasons:
>
>1. Faster convergence: SGD updates the model parameters more frequently than batch gradient descent, which leads to faster convergence to the optimal solution. This is because the updates are based on a smaller subset of the data, making them more computationally efficient.
>2. Reduced memory usage: Since SGD uses only a subset of the training data at each iteration, it requires less memory than batch gradient descent, which needs to load the entire dataset into memory.
>3. Avoids local minima: SGD's random sampling of training data at each iteration can help the algorithm avoid getting stuck in local minima. By randomly sampling the training data, SGD can escape from a local minimum and converge to a better solution.
>4. Works well for large datasets: SGD is more scalable and works well for large datasets that cannot fit into memory. By using small batches of data at each iteration, SGD can handle datasets of any size.
>5. Online learning: SGD is well-suited for online learning scenarios where the data is continuously arriving in a stream. With each new data point, SGD updates the model parameters, allowing it to adapt to new data in real-time.
>
>Overall, SGD's advantages make it a popular optimization algorithm in the field of machine learning, especially for large-scale and online learning scenarios.

SGD 对模型有更快的训练收敛表现、能减少内存使用、降低过拟合、适用于大/超大规模数据集或需要在线支持的模型训练。

## In what situations we need BGD but not SGD

>**AI:**
>
>There are some situations where batch gradient descent might be preferred over SGD:
>
>1. Small datasets: In cases where the dataset is relatively small, batch gradient descent might be a better choice since the entire dataset can be loaded into memory and processed in one go.
>2. Stable gradients: If the cost function has a smooth and stable gradient, then batch gradient descent might be preferable since it computes the gradient using the entire dataset, which provides a more accurate estimate of the direction of steepest descent.
>3. High-quality hardware: If the hardware used for training is powerful enough to handle large datasets efficiently, then batch gradient descent might be a better option since it can make use of the full power of the hardware.
>4. Time constraints: If there is no time constraint, batch gradient descent might be preferred since it guarantees convergence to the global minimum, whereas SGD only converges to a local minimum.
>5. Deterministic behavior: If the behavior of the algorithm needs to be deterministic and reproducible, then batch gradient descent might be preferred since it uses the same dataset at each iteration, whereas SGD uses a random subset of the dataset.
>
>In summary, batch gradient descent might be preferred over SGD in situations where the dataset is small, the cost function has stable gradients, the hardware is powerful, there are no time constraints, and deterministic behavior is desired.

BGD 并非一无是处，在小数据集、平稳梯度的训练场景中比较好用，如果需要可严格复现的训练过程、不计较时间成本的情况下，只能采用 BGD，因为 SGD 总是会随机地收敛在不同的局部最小值。

