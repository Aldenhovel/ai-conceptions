## Vision Transformer

## 简介

一个 `[32, 3, 224, 224]` 的批图像，经过patching得到 `[32, 16, 3, 32, 32]` 的 patches 输入到 ViT 中，首先展平为 `[32, 16, 3072]` ，然后经过 projection 层映射为隐向量 `[32, 16, 512]` 然后位置编码。在位置编码过后于第二维开头增加一个分类位得到 `[32, 17, 512]` ，经过 Encoder 还是相同形状，然后将分类位取出 `[32, 1, 512]` 转化为 `[32, 512]` 经过线性分类头得到 `[32, 10]` 为输出。

## 代码

>**NB:**
>
>```python
>import torch
>from torch import nn
>import math
>
>class VisionTransformer(nn.Module):
>    def __init__(self, patch_size, d_model, nhead, num_layers, num_classes):
>        super().__init__()
>        self.patch_size = patch_size
>        self.d_model = d_model
>
>        # 定义投影层
>        self.projection = nn.Linear(patch_size*patch_size*3 ,d_model)
>
>        # 定义类别嵌入
>        self.class_token = nn.Parameter(torch.randn(1 ,1 ,d_model))
>
>        # 定义位置编码
>        position=torch.arange(0 ,10000).unsqueeze(1)
>        div_term=torch.exp(torch.arange(0 ,d_model ,2) * -(math.log(10000.0) / d_model))
>        pos_enc=torch.zeros(1 ,10000 ,d_model)
>        pos_enc[0,:,0::2]=torch.sin(position * div_term)
>        pos_enc[0,:,1::2]=torch.cos(position * div_term)
>        
>        self.register_buffer('pos_enc',pos_enc)
>
>        # 定义transformer encoder
>        encoder_layer=nn.TransformerEncoderLayer(d_model,nhead)
>        self.transformer_encoder=nn.TransformerEncoder(encoder_layer,num_layers)
>
>        # 定义分类器
>        self.classifier=nn.Linear(d_model,num_classes)
>
>    def forward(self, patches):
>        batch_size,num_patches,c,h,w=patches.shape
>
>        # 展平patch向量
>        patches=patches.view(batch_size,num_patches,-1)
>
>        # 使用投影层转换patch向量
>        patches=self.projection(patches)
>
>        # 添加类别嵌入
>        class_token=self.class_token.expand(batch_size ,-1 ,-1)
>        patches=torch.cat([class_token ,patches] ,dim=1)
>
>        # 添加位置编码
>        patches+=self.pos_enc[:, :patches.shape[1], :]
>
>        # 输入到transformer encoder中进行进一步处理
>        output=self.transformer_encoder(patches.transpose(0 ,1)).transpose(0 ,1)
>
>        # 提取类别嵌入并输入到分类器中获得最终结果
>        logits=self.classifier(output[:, 0])
>
>        return logits
>
># 假设我们已经得到了图像batch的patches张量（shape: [batch_size,num_patches,c,h,w]）
>patches=torch.randn(32,16,(3),(32),(32))
>
># 创建VisionTransformer实例
>model=VisionTransformer(patch_size=32 ,
>                          d_model=512 ,
>                          nhead=8 ,
>                          num_layers=12 ,
>                          num_classes=10)
>
># 将patches张量输入到VisionTransformer中获得最终结果（shape: [batch_size,num_classes]）
>logits=model(patches)
>```
>
>在这里 x 是原始图像 `[32, 3, 224, 224]` 经过 patching 后得到的特征向量（还没有扁平化和经过位置编码），详见 [Image Patching]()。

