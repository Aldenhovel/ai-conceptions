# Tokenize

<img src="img/1.svg" alt="img" style="zoom:33%;" />

由上图， NLP 中由语句转化为训练模型数据时经过的中间环节，需要经过：

1.  **Normalization** 统一字母等。
2.  **Pre-tokenization** 划分成词级 `token` 。
3.  **Model-tokenization** 划分成语义级 `token` 。
4.  **Post-processor** 加上头尾 `CLS`  `SEP` 。

此外，在驶入模型前还需要：

5.  **Padding** 填充词序列到统一长度。
6.  **Covert to ids** 将词 `token` 根据词表转化为数值 `token` 。

## Normalization

此步比较简单，就是对基本字符的归一化，在英语中可以体现为大小写统一，在中文里体现为简繁体统一，其他语言也是相似道理。

```
["Hello how are U tday?"] --> ["hello how are u tday?"]
```

## Pre-tokenization

此步是字面意思的分词，一般就是按照空格将句子划分为词序列，即 `token` 。

```
["hello how are u tday?"] --> ["hello", "how", "are", "u", "tday", "?"]
```

## Model-tokenization

因为各种语言的不同，即使是相同含义的语句在词层面表述还是有点粗糙，例如 `play` 和 `playing` 按照分词应该属于两个完全不同的 `token` ，在没有先验知识去情况下，这两种应该看不出任何关系。有没有办法可以改进这种情况，答案是有的，可以根据语料库训练一个模型，让分词从词级进一步扩展到语义级，例如将 `playing` 划分成 `["play", "#ing"]` 。

需要注意的是这里我们使用的模型，与深度学习模型不同，这里训练 `Tokenizer` 使用的是固定算法构造映射关系，因此结果只取决于语料库内容，每次训练出来结果都是一样的（所以我们可以根据 `tokenizer` 设计模型，也可以根据模型设计 `tokenizer` 只要它们相匹配，所以经常在同一类细分任务下， `tokenizer` 可能是通用的）。

```
["hello", "how", "are", "u", "tday", "?"] --> ["hello", "how", "are", "u", "td", "##ay", "?"]
```

## Post-processor

至少在机器看来，一句话还需要有一个开始标志和结束标志。

```
["hello", "how", "are", "u", "td", "##ay", "?"] --> [<CLS>, "hello", "how", "are", "u", "td", "##ay", "?", "<SEP>"]
```

## Padding

语句长度不一是无法进行模型训练的，因此需要将不同长度的语句 `token` 序列填充到固定长度（超过的就截取），在填充时我们有一个专门的填充符号 `<pad>` ，填充后也会有一个 `mask` 序列告诉你哪些位置是原有 `token` 哪些是填充的 `token`。

```
[<CLS>, "hello", "how", "are", "u", "td", "##ay", "?", "<SEP>"] -->
{
	"tokens": [<CLS>, "hello", "how", "are", "u", "td", "##ay", "?", "<SEP>", "<PAD>"],
	"mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
}
```

## Convert to ids

这里把语句的 `token` 转化为 `id` 的形式。在前面我们训练 Model-tokenization 的时候，模型除了分词的映射关系之外，还会返回一个词表，例如 `token: "hello"` 对应 `id: 945` 这样，通过这个词表就可以将词转化为数字了。

```
[<CLS>, "hello", "how", "are", "u", "td", "##ay", "?", "<SEP>", "<PAD>"] --> 
[ 101,     8667,  1293,  1132, 158,  189,   6194, 136,     102,       0]
```

## 代码

这里介绍用 `transformer` 库的 `AutoTokenizer` 来加载已经训练好的 `tokenizer` ，这是因为重新训练一个 `tokenizer` 需要的语料库是比较庞大的，用现成的、经过实践检验的 `tokenizer` 岂不是更好更省事？

```python
# 使用 bert 的预训练 tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
```

`tokenizer` 有两种使用方式， **第一种方法** 是直接对语句进行 `__call__` 方法。可以一步到位完成以上全部操作：

```python
sequence = "Using a Transformer network is simple"

# 设置返回形式是 torch.Tensor ，最大长度是 10 ，使用最大长度填充
tokens = tokenizer(sequence, return_tensors="pt", max_length=10, padding="max_length")
print(tokens)
```

```
>>
{'input_ids': tensor([[  101,  7993,   170, 13809, 23763,  2443,  1110,  3014,   102,     0]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])}
```

**第二种方法** 是检查用的，将分割 `token` 和转化为 `id` 分开操作：

```python
sequence = "Using a Transformer network is simple"

# 分割
tokens = tokenizer.tokenize(sequence)

#转化
tokens_id = tokenizer.convert_tokens_to_ids(tokens)
print(tokens, tokens_id)
```

```
>>
(['Using', 'a', 'Trans', '##former', 'network', 'is', 'simple'],
 [7993, 170, 13809, 23763, 2443, 1110, 3014])
```

**注意，这种分开两步的操作不会自动生成头尾标识符，也不会自动填充到固定长度，比较适合检查时用。**

反之将 `token` 转化回语句也是 `tokenizer` 的工作之一：

```python
tokenizer.decode([7993, 170, 13809, 23763, 2443, 1110, 3014])
```

```
>>
'Using a Transformer network is simple'
```

## 封装

因为 NLP 的下游任务也有不少差别， 不同`tokenizer` 本身功能也有一定差别，最好根据自己需求重新封装一下，以下代码是对 `BERTTokenizer` 的一个重新封装：

```python
from transformers import AutoTokenizer

class BERTTokenizer:
    def __init__(self, max_len):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.max_len = max_len
        
    def encode(self, sentence):
        return self.tokenizer(sentence, max_length=self.max_len, padding="max_length", return_tensors="pt")
    
    def decode(self, ids):
        return self.tokenizer.decode(ids)
    
    def split_tokens(self, sentence):
        return self.tokenizer.tokenize(sentence)
    
    def encode_tokens(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)
```

相似，如果要用 `GPT` 或者其他的 `tokenizer` ，也最好重新写一个，将需要的功能抽出来做出函数。

