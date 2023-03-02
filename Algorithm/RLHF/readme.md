# Reinforcement Learning from Human Feedback (RLHF)

## 技术分解

字面上说，RLHF就是基于人类反馈（Human Feedback）对语言模型进行强化学习（Reinforcement Learning），和一般的fine-tune过程乃至prompt tuning自然也不同。RLHF 是一项涉及多个模型和不同训练阶段的复杂概念，这里我们按三个步骤分解：

1. 预训练一个语言模型 (LM) ；
2. 聚合问答数据并训练一个奖励模型 (Reward Model，RM) ；
3. 用强化学习 (RL) 方式微调 LM。

## 微调预训练模型

花钱招人给问题（prompt）写回答（demonstration），然后finetune一个GPT3。这一步大家都懂，就不用说了。这一步可以多训几个版本，第二步会用到。

![img](https://pic3.zhimg.com/80/v2-fb3e3120c6d50ef09c7b2601a26241d2_720w.webp)

## 训练奖励模型

用多个模型（可以是初始模型、finetune模型和人工等等）给出问题的多个回答，然后人工给这些问答对按一些标准（可读性、无害、正确性）进行排序，训练一个奖励模型/偏好模型来打分（reward model）。

![img](https://pic1.zhimg.com/80/v2-b22f4564d13c54cf27c80d90da622170_720w.webp)

>Q1：为什么不人工直接打分？
>
>A1：因为打分是主观的需要归一化，而排序一般大家会有共同的结论：对同一个问题，A和B哪个回答更好。
>
>Q2：有了一组一组的偏序（A>B, A>C, C>B）怎么得到每个回答的奖励分数？
>
>A2：在Hug的博客里用了Elo排名系统。
>
>Q3：这个RM用什么模型？
>
>A3：只要用Elo系统打分后归一化，然后直接上个LM做回归就行，可以从零训练也可以用老LM做finetune。这里有个有趣的事情在于，做问答和做评分都需要输入所有的文本，实际上两个模型的容量（或者说理解能力）应该是差不多的，而现有的RLHF模型都使用了两个**不同**大小的模型。
>
>Q4：有没有其他方式训练打分的模型？
>
>A4：张俊林老师指出对偏序直接用pairwise learning to rank做打分，大概更符合常规的思路 [ChatGPT会取代搜索引擎吗](https://zhuanlan.zhihu.com/p/589533490) 。

## 强化学习

用强化学习训练上面那个 finetune 后的 GPT3 模型。用强化学习做 LM 训练的一种思路是用Policy Gradient做，这一块 OpenAI 用的是他们在17年提出的 PPO 算法，即Proximal Policy Optimization。

![img](https://pic4.zhimg.com/80/v2-2a097d5661209c81476fdd87be89d95f_720w.webp)

## 参考文献

[从零实现ChatGPT——RLHF技术笔记](https://zhuanlan.zhihu.com/p/591474085)

[Hugging Face](https://huggingface.co/blog/rlhf)

[Hugging Face - 简中](https://mp.weixin.qq.com/s/TLQ3TdrB5gLb697AFmjEYQ)

