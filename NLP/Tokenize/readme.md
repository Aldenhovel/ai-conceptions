# Tokenize

NLP 中由语句转化为训练模型数据时经过的中间环节，需要：

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

需要注意的是这里我们使用的模型，与深度学习模型不同，这里训练 `Tokenizer` 使用的是固定算法构造映射关系，因此结果只取决于语料库内容，每次训练出来结果都是一样的。正是这样的与深度学习无关的特性，我们可以将同一个 `Tokenizer` 用在不同的 NLP 模型上。

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

