# Resnet-BahdanauAttention-LSTM Model

## Resnet Encoder

```python
from torchvision.models import resnet50, ResNet50_Weights
class ResNetEncoder(nn.Module):
    
    def __init__(self, hidden_size=2048):
        super(ResNetEncoder, self).__init__()
        #resnet = torchvision.models.wide_resnet101_2(pretrained=True)
        resnet = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-2]
        self.hidden_size = hidden_size
        self.resnet = nn.Sequential(*modules)
        
    def fine_tuning_resnet(self, fine_tune):
        for p in self.resnet.parameters():
            p.requires_grad = False
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune
        
    def forward(self, images):
        """
        :param
            images: Tensor[batch_size, 3, img_size, img_size]
        :return
            out: Tensor[batch_size, 8, 8, hidden_size]
        """
        out = self.resnet(images)
        out = out.permute(0, 2, 3, 1)
        return out
```

```python
def test_encoder():
    encoder = ResNetEncoder()
    latent = encoder(torch.rand((10, 3, 256, 256)))
    assert latent.size() == torch.Size([10, 8, 8, 2048]), latent.size() 
    
test_encoder()
```

## BadhanauAttention

```python
class BahdanauAttention(nn.Module):
    
    def __init__(self, enc_hidden_size, dec_hidden_size, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(enc_hidden_size, hidden_size)
        self.W2 = nn.Linear(dec_hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, features, h_state):
        """
        :param
            features:  Tensor[batch_size, num_pixels, enc_hidden_size]
            h_state: Tensor[batch_size, dec_hidden_size]
        :return
            context_vector: Tensor[batch_size, enc_hidden_size]
            attention_weights: Tensor[batch_size, num_pixels]
        """
        h_state = h_state.unsqueeze(1) 
        # [batch_size, 1, dec_hidden_size]
        score = F.elu(self.W1(features) + self.W2(h_state)) 
        # [batch_size, num_pixels, hidden_size]
        attention_weights = F.softmax(self.V(score), dim=1) 
        # [batch_size, num_pixels, 1]
        context_vector = attention_weights * features 
        # [batch_size, num_pixels, enc_hidden_size]
        context_vector = torch.sum(context_vector, dim=1) 
        # [batch_size, enc_hidden_size]
        return context_vector, attention_weights.squeeze(2)
```

```python
def test_attention():
    attention = BahdanauAttention(enc_hidden_size=2048, dec_hidden_size=512, hidden_size=512)
    context_vector, attention_weights = attention(torch.rand((10, 8*8, 2048)), torch.rand((10, 512)))
    assert context_vector.size() == torch.Size([10, 2048])
    assert attention_weights.size() == torch.Size([10, 8*8])
    
test_attention()
```

## LSTM Decoder

```python
class DecoderWithBahdanauAttention(nn.Module):
    
    def __init__(self, enc_hidden_size, attn_hidden_size, hidden_size, embedding_size, vocab_size, dropout):
        super(DecoderWithBahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(embedding_size + enc_hidden_size, hidden_size)
        self.attention = BahdanauAttention(enc_hidden_size, 
                                           hidden_size, 
                                           attn_hidden_size)
        self.f_beta = nn.Linear(hidden_size, enc_hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)
        
    def fine_tuning_embeddings(self, fine_tune=False):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune
        
    def forward(self, input_word_index, h_state, c_state, enc_outputs):
        """
        :param
            input_word_index: Tensor[batch_size,]
            h_state: Tensor[1, batch_size, hidden_size]
            c_state: Tensor[1, batch_size, hidden_size]
            enc_outputs: Tensor[batch_size, num_pixels, enc_hidden_size]
        :return
            logit: Tensor[batch_size, vocab_size]
            h_state: Tensor[1, batch_size, hidden_size]
            c_state: Tensor[1, batch_size, hidden_size]
            attention_weights: Tensor[batch_size, num_pixels]
        """
        embedded = self.embedding(input_word_index)  
        # embbeded: [batch_size, embedding_size]
        context_vector, attention_weights = self.attention(enc_outputs,
                                                           h_state.squeeze(0))
        # context_vector: Tensor[batch_size, enc_hidden_size]
        # attention_weights: Tensor[batch_size, num_pixels]
        
        gate = torch.sigmoid(self.f_beta(h_state))  
        # [1, batch_size, enc_hidden_size], Gating scalar
        context_vector = gate.squeeze(0) * context_vector 
        # [batch_size, enc_hidden_size]
        
        x = torch.cat((embedded, context_vector), dim=1) 
        # [batch_size, embedding_size + enc_hidden_size]
        output, (h_state, c_state) = self.lstm(x.unsqueeze(0), (h_state, c_state))
        # output: [1, batch_size, hidden_size]
        # h_state: [1, batch_size, hidden_size]
        # c_state: [1, batch_size, hidden_size]
        
        logit = self.fc(self.dropout(output)) # [1, batch_size, vocab_size]
        logit = logit.squeeze(0) # [batch_size, vocab_size]
        return logit, h_state, c_state, attention_weights
```

## model

```python
class AutoEncoder(nn.Module):
    
    def __init__(self, encoder, decoder, device):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.init_h0 = nn.Linear(encoder.hidden_size, decoder.hidden_size)
        self.init_c0 = nn.Linear(encoder.hidden_size, decoder.hidden_size)
        self.device = device
        
    def forward(self, images, target_sequences, sequence_lengths, tf_ratio):
        """
        :param
            images: Tensor[batch_size, 3, img_size, img_size]
            target_sequences: Tensor[batch_size, seq_len]
            sequence_lengths: Tensor[batch_size,]
            tf_ratio: float
        :return
            logits: Tensor[max(decode_lengths), batch_size, vocab_size]
            logits: Tensor[batch_size, max(decode_lengths), num_pixels]
            sorted_target_sequences: Tensor[seq_len, batch_size]
            sorted_decode_lengths: list[seq_len]
            sorted_indices: list[batch_size]
        """
        batch_size = images.size(0)
        
        # Encoding
        image_features = self.encoder(images) 
        # [batch_size, 14, 14, hidden_size]
        image_features = image_features.view(batch_size, -1, self.encoder.hidden_size) 
        # [batch_size, num_pixels, enc_hidden_size]
        num_pixels = image_features.size(1)
        
        # Sort the batch by decreasing lengths
        sorted_sequence_lengths, sorted_indices = torch.sort(sequence_lengths, 
                                                             dim=0, 
                                                             descending=True)
        sorted_image_features = image_features[sorted_indices] 
        # [batch_size, num_pixels, enc_hidden_size]
        sorted_target_sequences = target_sequences[sorted_indices] 
        # [seq_len, batch_size]
        
        ## Init hidden and memory states
        mean_image_features = sorted_image_features.mean(dim=1) 
        # [batch_size, enc_hidden_size]
        h_state = self.init_h0(mean_image_features)
        c_state = self.init_c0(mean_image_features) 
        # [batch_size, dec_hidden_size]
        h_state, c_state = h_state.unsqueeze(0), c_state.unsqueeze(0) 
        # [1, batch_size, dec_hidden_size]
        
        ## We won't decode at the <eos> position, 
        ## since we've finished generating as soon as we generate <eos>
        ## So, decoding lengths are actual lengths - 1
        ## 这里将语句从长到短排序，表示每个每个句子要解码的长度，
        ## 减1是因为最后一个 <eos> 是不需要解码的，因此所有语句解码长度都减1
        sorted_decode_lengths = (sorted_sequence_lengths - 1).tolist()
        
        # Decoding
        ## 这里的 logits 对应每个时间步的解码结果， alphas 是 abhdanau attn 里面的 alpha
        ## 因此 logits.shape 对应 (seq_len, B, hidden_size) 用于 LSTM 输入
        ## alphas.shape 对应 (B, seq_len, num_pixels) 对应每个词和 CNN 特征图的注意力
        logits = torch.zeros(max(sorted_decode_lengths), 
                             batch_size, 
                             self.decoder.vocab_size
                            ).to(self.device)
        alphas = torch.zeros(batch_size, 
                             max(sorted_decode_lengths), 
                             num_pixels
                            ).to(self.device)
        
        
        last = None
        for t in range(max(sorted_decode_lengths)):
            batch_size_t = sum([l > t for l in sorted_decode_lengths])
            
            # 这里实际上等价于
            if last is not None:
                ## tf_ratio=0，这一步不会发生
                if random.random() < tf_ratio:
                    in_ = last[:batch_size_t]
                else:
                    in_ = sorted_target_sequences[:batch_size_t, t]
            else:
                in_ = sorted_target_sequences[:batch_size_t, t]
            
            logit, h_state, c_state, attention_weights \
            = self.decoder(in_,
                           h_state[:, :batch_size_t, :],
                           c_state[:, :batch_size_t, :],
                           sorted_image_features[:batch_size_t, :, :])
            
            
            logits[t, :batch_size_t, :] = logit
            alphas[:batch_size_t, t, :] = attention_weights
            last = torch.argmax(F.softmax(logit, dim=1), dim=1) 
            # [batch_size,]
        
        return logits, \
                alphas, \
                sorted_target_sequences, \
                sorted_decode_lengths, \
                sorted_indices
```

