# Low-rank Adaptation (LoRA)

## 简介

LoRA，英文全称Low-Rank Adaptation of Large Language Models，直译为大语言模型的低阶适应，这是微软的研究人员为了解决大语言模型微调而开发的一项技术。

目前大语言模型在针对特定任务时一般采用预训练-微调方式，但对多数 LLM 来说，如 GPT-3 有数十亿参数，它能微调，但成本太高太麻烦了。LoRA的做法是，冻结预训练好的模型权重参数，然后在每个Transformer 块里注入可训练的层，就好比是大模型的一个小模型或者说是一个插件。由于不需要对模型的权重参数重新计算梯度，所以大大减少了微调需要训练的计算量。

## 用途

LoRA 本来是给 LLM 准备的，但把它用在cross-attention layers（交叉关注层）也能影响用文字生成图片的效果。因此在一些 text-to-image 模型上也有它的身影，例如 Stable Diffusion 模型 [Low-rank Adaptation for Fast Text-to-Image Diffusion Fine-tuning](https://github.com/cloneofsimo/lora) ，见下图：

![潜在扩散结构](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/lora-assets/latent-diffusion.png)

## hugging face 支持

更多相关信息请看 Hugging Face 的博客主页 [传送门](https://huggingface.co/blog/lora) 。

