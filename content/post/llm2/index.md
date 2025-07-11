---
title: 手搓Transformer：深入架构细节
description: 大模型学习笔记（二）
slug: llm2
date: 2025-06-03 22:05:15+0800
math: true
image: img/cover.png
categories:
    - 文档
    - AI Infra
tags:
    - 文档
    - AI Infra
weight: 2
---

## Self-Attention

### SDPA

$$Attention(Q, K, V)=softmax(\frac{QK^{T}}{\sqrt[]{d_{k}} })V$$

根据计算公式可以得到Attention的计算流程，

+ 首先计算`Attention Score`：将$q$和$k^T$批量矩阵乘（BMM, Batch Matrix Multiplication）并除以Scaled因子，如果是Masked Self-Attention则需要通过掩码对mask为0的位置替换为`-inf`(`exp(-inf)=0`)

+ 对`Attention Score`在行维度上softmax后与$v$批量矩阵乘，得到Attention的输出

```python
def scaled_dot_product_attention(query, key, value, mask=None):
    dim_k = key.size(-1)
    attn_scores = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(dim_k)
    if mask is not None:
        attn_scores.masked_fill_(mask == 0, float('-inf'))
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_outputs = torch.bmm(attn_weights, value)
    return attn_outputs
```

同时，PyTorch也提供了一个[Efficient的SDPA算子](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)，在矩阵规模较大时有一定加速效果：

```python
def scaled_dot_product_attention(query, key, value, mask=None):
    attn_output = F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False
    )
    return attn_output
```

SDPA是一个高度优化的算子，通过Python接口封装底层C++/CUDA实现，下面是Python的接口调用：

```python
# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value
```

### MHA

{{< figure src="img/1.png#center" width=600px" title="Self-Attention">}}

首先需要通过`AttentionHead`类实现一个单头注意力机制（SHA）作为MHA的组件，每个SHA会将`embed_dim`维度的信息映射到`head_dim`维度上：

```python
class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        # Learnable Parameters
        self.Wq = nn.Linear(embed_dim, head_dim)
        self.Wk = nn.Linear(embed_dim, head_dim)
        self.Wv = nn.Linear(embed_dim, head_dim)
        
    def forward(self, query_input, key_value_input):
        # Project Q
        q = self.Wq(query_input)
        # Project K
        k = self.Wk(key_value_input)
        # Project V
        v = self.Wv(key_value_input)
        attn_outputs = scaled_dot_product_attention(q, k, v)
        return attn_outputs
```

注意在Encoder Layer中，Self-Attention的q、k、v的输入都是同样的hidden states；但是在Decoder Layer中，q的输入是上一层hidden states，但是k、v的输入是来自最后一层Encoder Layer的hidden states，因此Attention Head如此设计。

接下来是MHA的实现，注意变量经过MHA维度是不会发生变化的（`embed_dim` -> `embed_dim`）：

$$MHA(Q, K, V) = concat(head_{1}, ..., head_{h})W^{O} \newline  
(where\ head_{i} = Attention(QW^{Q}_{i}, KW^{K}_{i}, VW^{V}_{i}))$$

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size          # 768
        num_heads = config.num_attention_heads  # 12
        head_dim = embed_dim // num_heads       # 64
        self.heads = nn.ModuleList([
            AttentionHead(embed_dim, head_dim) for _ in range(num_heads)
        ])
        # 768 -> 768
        self.output_layer = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, query_input, key_value_input, mask=None):
        x = torch.cat([head(query_input, key_value_input, mask) for head in self.heads], dim=-1)
        x = self.output_layer(x)
        x = self.dropout(x)
        return x
```

## Transformer Encoder Layer

{{< figure src="img/2_.png#center" width=300px" title="Transformer Encoder Layer">}}

### Feed Forward Network

Transformer架构的FFN使用GeLU（Gaussian Error Linear Unit）激活函数，可以看作是ReLU的平滑版本，具体公式如下：

$$GeLU(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2} (1 + \text{erf}(\frac{x}{\sqrt{2}}))$$

其中，

+ $\Phi(x)$是标准高斯分布的累计分布函数（CDF）

+ $erf(x)$是误差函数，定义为$\text{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2}$

{{< figure src="img/3_.png#center" width=400px" title="Activation Function Comparison">}}

接下来实现FFN，具体为两层MLP和一次GeLU激活：

```python
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        # (intermediate)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # (output)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
```

### Add & Layer Norm

这里直接将Add & Layer Norm写到最后的TransformerEncoderLayer中，一般取`layer_norm_eps=1e-12`，这里给出Post-LN（Transformer原文）的实现：

```python
# Post-LN
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.layernorm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states):
        attn_output = self.attention(hidden_states, hidden_states)
        hidden_states = self.layernorm1(hidden_states + attn_output)
        
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.layernorm2(hidden_states + ffn_output)
        
        return hidden_states
```

{{< figure src="img/4.png#center" width=400px" title="Comparison between Post-LN and Pre-LN">}}

## Transformer Decoder Layer

{{< figure src="img/5.png#center" width=300px" title="Transformer Decoder Layer">}}

有了前面Encoder Layer的实现，Decoder Layer也就能够跟着架构图水到渠成了：

```python
class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.cross_attn = MultiHeadAttention(config)
        self.ffn = FeedForward(config)

        self.layernorm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm3 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, encoder_outputs, self_attn_mask=None, cross_attn_mask=None):
        self_attn_output = self.self_attn(hidden_states, hidden_states, self_attn_mask)
        hidden_states = self.layernorm1(hidden_states + self_attn_output)

        cross_attn_output = self.cross_attn(hidden_states, encoder_outputs, cross_attn_mask)
        hidden_states = self.layernorm2(hidden_states + cross_attn_output)

        ffn_output = self.ffn(hidden_states)
        hidden_states = self.layernorm3(hidden_states + ffn_output)

        return hidden_states
```

## Source Code

完整测试Demo如下：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def scaled_dot_product_attention(query, key, value, mask=None):
    dim_k = key.size(-1)
    attn_scores = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(dim_k)
    if mask is not None:
        attn_scores.masked_fill_(mask == 0, float('-inf'))
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_outputs = torch.bmm(attn_weights, value)
    return attn_outputs

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        # Learnable Parameters
        self.Wq = nn.Linear(embed_dim, head_dim)
        self.Wk = nn.Linear(embed_dim, head_dim)
        self.Wv = nn.Linear(embed_dim, head_dim)
        
    def forward(self, query_input, key_value_input):
        # Project Q
        q = self.Wq(query_input)
        # Project K
        k = self.Wk(key_value_input)
        # Project V
        v = self.Wv(key_value_input)
        attn_outputs = scaled_dot_product_attention(q, k, v)
        return attn_outputs

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size          # 768
        num_heads = config.num_attention_heads  # 12
        head_dim = embed_dim // num_heads       # 64
        self.heads = nn.ModuleList([
            AttentionHead(embed_dim, head_dim) for _ in range(num_heads)
        ])
        # 768 -> 768
        self.output_layer = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, query_input, key_value_input, mask=None):
        x = torch.cat([head(query_input, key_value_input, mask) for head in self.heads], dim=-1)
        x = self.output_layer(x)
        x = self.dropout(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        # (intermediate)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # (output)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.layernorm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states):
        attn_output = self.attention(hidden_states, hidden_states)
        hidden_states = self.layernorm1(hidden_states + attn_output)
        
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.layernorm2(hidden_states + ffn_output)
        
        return hidden_states

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.cross_attn = MultiHeadAttention(config)
        self.ffn = FeedForward(config)

        self.layernorm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm3 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, encoder_outputs, self_attn_mask=None, cross_attn_mask=None):
        self_attn_output = self.self_attn(hidden_states, hidden_states, self_attn_mask)
        hidden_states = self.layernorm1(hidden_states + self_attn_output)

        cross_attn_output = self.cross_attn(hidden_states, encoder_outputs, cross_attn_mask)
        hidden_states = self.layernorm2(hidden_states + cross_attn_output)

        ffn_output = self.ffn(hidden_states)
        hidden_states = self.layernorm3(hidden_states + ffn_output)

        return hidden_states

class DummyConfig:
    hidden_size = 768
    num_attention_heads = 12
    intermediate_size = 3072
    hidden_dropout_prob = 0.1
    layer_norm_eps = 1e-12

if __name__ == '__main__':
    config = DummyConfig()

    batch_size = 2
    seq_len = 10
    hidden_size = config.hidden_size

    # 输入张量
    dummy_input = torch.randn(batch_size, seq_len, hidden_size)

    # 测试 Encoder Layer
    encoder_layer = TransformerEncoderLayer(config)
    encoder_output = encoder_layer(dummy_input)
    print("Encoder Output Shape:", encoder_output.shape)

    # 测试 Decoder Layer
    decoder_layer = TransformerDecoderLayer(config)
    decoder_input = torch.randn(batch_size, seq_len, hidden_size)
    decoder_output = decoder_layer(decoder_input, encoder_output)
    print("Decoder Output Shape:", decoder_output.shape)
```

Outputs:

```text
Encoder Output Shape: torch.Size([2, 10, 768])
Decoder Output Shape: torch.Size([2, 10, 768])
```

---

## Reference

[On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)

[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

[Document: torch.nn.functional.scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

[bilibili-五道口纳什-BERT、T5、GPT合集](https://search.bilibili.com/all?vt=69025821&keyword=%E4%BA%94%E9%81%93%E5%8F%A3%E7%BA%B3%E4%BB%80&from_source=webtop_search&spm_id_from=333.1007&search_source=5)