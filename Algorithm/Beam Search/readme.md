# Beam Search 束搜索

<img src="img/1.png" alt="img" style="zoom: 33%;" />

这里是一个 `beam_size=2` 的Beam Search示意图，每个节点都会扩展5个下级节点，在 Beam Search 每次都会从所有扩展节点里面挑选出2个累计启发值最大的节点，直到达到结束标准。

## 理念

Beam Search 是对 Greedy Search（贪心搜索）的一个改进算法，能够扩展贪心搜索的搜索空间。

以语言生成为例，指定 `beam_size=k` ，Beam Search 的算法描述为：

1. 将开始节点 $x_0$ 通过模型预测生成 $m$ 个节点 $\hat{x}_1 :[m, ]$
2. 从 $\hat{x}_1$ 中挑选出概率最大的 $k$ 个节点 $x_1:[k, ]$
3. 将 $x_1$ 每个节点通过模型预测生成 $k\times m$ 个节点 $\hat{x}_2:[k \times m,]$
4. 从 $\hat{x}_2$ 中挑选出概率最大的 $k$ 个节点 $x_2:[k, ]$
5. ......
6. 直到 $k$ 个语句全部遇到 `<eos>` 或者超过最大搜索深度。

## 示例

```python
from math import log
from numpy import array
from numpy import argmax

# beam search
def beam_search_decoder(data, k):
    sequences = [[list(), 1.0]]
    for row in data:
        all_candidates = list()
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score * -log(row[j])]
                all_candidates.append(candidate)
        # 所有候选根据分值排序
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        # 选择前k个
        sequences = ordered[:k]
    return sequences

# 定义一个句子，长度为10，词典大小为5
data = [[0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1]]
data = array(data)
# 解码
result = beam_search_decoder(data, 3)
# print result
for seq in result:
    print(seq)
```

这是一个 Beam Search 例子，假设我们已经计算出语句每个位置单词的启发值（这里例子和真实的 NLP情况有所区别，真实的 NLP 里词是一个个产生的，不能一次产生全部）。这样可以得到 Beam Search 搜索结果，输出3个期望最高的语句。

