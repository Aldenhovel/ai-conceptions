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
        h_state = h_state.unsqueeze(1) # [batch_size, 1, dec_hidden_size]
        score = F.elu(self.W1(features) + self.W2(h_state)) # [batch_size, num_pixels, hidden_size]
        attention_weights = F.softmax(self.V(score), dim=1) # [batch_size, num_pixels, 1]
        context_vector = attention_weights * features # [batch_size, num_pixels, enc_hidden_size]
        context_vector = torch.sum(context_vector, dim=1) # [batch_size, enc_hidden_size]
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

