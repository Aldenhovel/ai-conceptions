## DETR

## 代码

```python
class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim, nheads, num_encoder_layers, num_decoder_layers):
        nn.Module.__init__(self)
        # backbone 取 resnet50 的骨干，除去最后两层（avgpool 和 fc），用来做特征提取
        self.backbone = nn.Sequential(
            *list(resnet50(pretrained=True).children())[:-2])
        # conv 是 kernel_size=1 ，将 2048 维压缩成 hidden_size 维
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        # transformer 层
        self.transformer = nn.Transformer(hidden_dim, 
                                          nheads, 
                                          num_encoder_layers, 
                                          num_decoder_layers)
        # 将 transformer 出来的 (query_pos, hidden_dim) 转化为 (query_pos, num_classes)
        # 对应 query_pos 个物体的类别概率
        self.linear_class = nn.Linear(hidden_dim, num_classes)
        # 同理，转化为 query_pos 个 bbox 坐标高宽
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        # object query ，在 transformer 解码器端做输入
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        # 位置编码矩阵（行和列）， 这里使用了随机初始化的可学习位置编码
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
    
    def forward(self, inputs):
        print(f"inputs: {inputs.shape}")
        
        x = self.backbone(inputs)
        print(f"x: {x.size()}")
        
        h = self.conv(x)
        print(f"h: {h.size()}")
        
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        print(f"pos: {pos.shape}")
        
        print(f"h_flatten: {h.flatten(2).permute(2, 0, 1).shape}")
        print(f"objq: {self.query_pos.unsqueeze(1).permute(2, 0, 1).shape}")
        # 注意，在 nn.Transformer 里面 src 和 tgt 的输入是 (seq_len, batch, d_model)
        # 所以下面才会有 permute(2, 0, 1)，第一位不是 batch size !!
        h = self.transformer(pos + h.flatten(2).permute(2, 0, 1), 
                             self.query_pos.unsqueeze(1))
        print(f"h_decoded: {h.shape}")
        
        return self.linear_class(h), self.linear_bbox(h).sigmoid()
```

```python
detr = DETR(num_classes=91, 
            hidden_dim=256, 
            nheads=8,
            num_encoder_layers=6, 
            num_decoder_layers=6)
detr.eval()
inputs = torch.randn(1, 3, 300, 400)
logits, bboxes = detr(inputs)
print(f"classes: {logits.shape}, bboxes: {bboxes.shape}")
```

```
>>
inputs: torch.Size([1, 3, 300, 400])
x: torch.Size([1, 2048, 10, 13])
h: torch.Size([1, 256, 10, 13])
pos: torch.Size([130, 1, 256])
h_flatten: torch.Size([130, 1, 256])
objq: torch.Size([256, 100, 1])
h_decoded: torch.Size([100, 1, 256])
classes: torch.Size([100, 1, 91]), bboxes: torch.Size([100, 1, 4])
```

