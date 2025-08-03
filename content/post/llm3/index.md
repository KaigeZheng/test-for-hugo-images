---
title: Infra入门——An Overview of AI Infra
description: 大模型学习笔记（三）
slug: llm3
date: 2025-08-03 23:56:00+0800
math: true
image: img/cover.jpg
categories:
    - 文档
    - AI Infra
tags:
    - 文档
    - AI Infra
weight: 2
---

计划在这篇博客里调研并粗略地学习一下到目前为止比较有影响力的AI Infra工作（类似Survey），并慢慢补充丰富。Anyway，迈出行动的第一步最难。

## Model Parameters

### Parameter Estimation

1B = 1 Billion = 十亿

假设模型层数为$N$，隐藏层维度为$H$，接下来考虑一层Transformer层的参数量估算：

- 自注意力层：（不需要考虑MHA的情况，因为多头concat起来的为度就等于隐藏层维度）需要注意的是这里包括一次注意力计算和一次线性映射，因此涉及四个可训练参数$W_Q$、$W_K$、$W_V$和$W_O$，因此注意力层的可训练参数量为$4H^2 + 4H$。

$$Attention(Q, K, V)=softmax(\frac{QK^{T}}{\sqrt[]{d_{k}} })V \newline Q=W_{Q}X+b_{Q}, K=W_{K}X+b_{K}, V=W_{V}X+b_{V}, O = W_{O}Attention_{O}+b_{O}$$

- 前馈网络层：FFN层包括一次线性升维和一次线性降维，设计两个可训练参数$W_{1}$和$W_{2}$，可训练参数量为$(H\times 4H + 4H) + (4H \times H + H) = 8H^{2} + 5H$。

$$FFN(x) = GeLU(xW_{1}+b_{1})W_{2}+b_{2}$$

- 残差连接和层归一化：Add & Norm层（主要是LN层）涉及两个可训练向量参数$\alpha$和$b$，即2H。

$$Y = Attention(X) + X \newline LN(Y) = \alpha \frac{Y - \mu}{\sigma} + b$$

综上，一层Transformer层由一层Attention层、一层FFN层和两层Add & Norm层组成，可训练参数量为$12H^{2} + 13H$，可以近似为$12H^{2}$。

### Conputation Estimation

> AxB和BxC的矩阵相乘，每个输出元素需要进行$n$次乘法和$n-1$次加法，$\approx 2n FLOPs$；整个矩阵共有AxC个输出元素，因此总$FLOPs \approx 2ABC$，可以近似为$ABC$。因此估算参数时，主要关注矩阵乘或向量矩阵乘的维度即可。注意$W_{Q/K/V}$的维度是$HxH$，而$Q/K/V$的维度是$LxH$。

接下来估算计算量，由于LayerNorm、Dropout等计算量较小，暂时不考虑。设模型层数为$N$，隐藏层维度为$H$，批量大小为$B$，序列长度为$L$：

- 自注意力层

1. （LHxHH=LH）线性投影QKV：每个投影是$H\times H$，应用于每个token就是$BLH^{2}$，总共3个矩阵，因此$FLOPs=3BLH^{2}$

2. （LHxHL=LL）Attention Score ($QK^{T}$)：每个token对应一个$L\times L$的注意力矩阵，需要做$H$次乘加，约为$BHL^{2}$

3. （——————）Softmax和Scaling：Softmax涉及取指、求和、逐元素除和、数值稳定的计算，这里只能估计为$xBL^{2}$，相比SDPA可忽略不计

4. （LLxLH=LH）Attention Output与V相乘：显然是$BHL^{2}$

5. （LHxHH=LH）输出线性层$W_{O}$：显然是$BLH^{2}$

因此，自注意力层的总FLOPs：

$$\approx (3BLH^{2})+(2BHL^{2})+(BLH^{2})=4BLH^{2}+2BHL^{2}$$

- 前馈网络层：

1. 升维（$H$->$4H$）：$FLOPs=BLH(4H)$

2. 降维（$4H$->$H$）：$FLOPs=BL(4H)H$

（当然还有GeLU激活，不过是线性的计算量，可以忽略不计）因此，FFN层的总Flops：

$$\approx 8BLH^{2}$$

综上，$Total\ FLOPs \approx 12BLH^{2} + 2BHL^{2}$$。（理论上应该再乘以2）

## Memory Estimation



## Inference Optimization

### KV Cache OPtimization

#### KV Cache

KV (Key-Value) Cache是一种在自回归模型（如Decoder of Transformer）中常用的推理加速技术，通过在推理的注意力机制计算过程中缓存已计算过的$Key$和$Value$，减少重复的$K$、$V$与权重矩阵的projection计算。

$$Attention(Q, K, V)=softmax(\frac{QK^{T}}{\sqrt[]{d_{k}} })V$$

为什么可以缓存$K$和$V$？由于**Casual Mask**机制，当模型推理时当前token不需要与之后的token进行Attention计算，因此在计算第$t$个token的$Attention_{t}$时，只需要$Q_{0:t}$、$K_{0:t}$和$V_{0:t}$。而Decoder中的$Q$需要token在embedding后通过$W_q$投影，但$K_{0:t-1}$与$V_{0:t-1}$来自Encoder中，且在计算$Attention_{0:t-1}$时已被计算过，因此可以通过缓存已被计算过的历史$K$与$V$来节省这部分计算。

接下来参考[知乎@看图学](https://zhuanlan.zhihu.com/p/662498827)的公式推导，

计算第一个token时的Attention：

$$
Attention(Q, K, V) = softmax(\frac{Q_{1}K_{1}^{T}}{\sqrt[]{d}})V_{1}
$$

计算第二个token时的Attention（矩阵第二行对应$Attention_{2}$），$softmax(\frac{Q_{1}K_{2}}{\sqrt d})$项被mask掉了：

$$
Attention(Q, K, V) = softmax(\frac{Q_{2}[K_{1}, K_{2}]^{T}}{\sqrt[]{d}})[V_{1}, V_{2}] \newline = \begin{pmatrix}
softmax(\frac{Q_{1}K_{1}^{T}}{\sqrt d})  & softmax(-\infty )\\
softmax(\frac{Q_{2}K_{1}^{T}}{\sqrt d})  & softmax(\frac{Q_{2}K_{2}^{T}}{\sqrt d})
\end{pmatrix}[V_{1}, V_{2}] \newline =\begin{pmatrix}
softmax(\frac{Q_{1}K_{1}^{T}}{\sqrt d})V_{1} + 0 \times V_{2} \\
softmax(\frac{Q_{2}K_{1}^{T}}{\sqrt d})V_{1} + softmax(\frac{Q_{2}K_{2}^{T}}{\sqrt d})V_{2}
\end{pmatrix}
$$

以此类推，Attention矩阵的严格上三角部分都被mask掉了，因此**计算第$t$个token的$Attention_{t}$时与$Q_{1:t-1}$无关**：

$$
Attention_{1} = softmax(\frac{Q_{1}K_{1}^{T}}{\sqrt[]{d}})V_{1} \newline Attention_{2} = softmax(\frac{Q_{1}[K_{1}, K_{2}]^{T}}{\sqrt[]{d}})[V_{1}, V_{2}]  \newline ... \newline Attention_{t} = softmax(\frac{Q_{t}K_{1:t}^{T}}{\sqrt[]{d}})V_{1:t}
$$

源码实现参考[Huggingface的GPT2推理实现](https://github.com/huggingface/transformers/blob/c962f1515e40521c0b336877a64dc512da4f486d/src/transformers/models/gpt2/modeling_gpt2.py#L269-L360)，KV Cache的逻辑核心思路如下：

- 对于`Cross Attention`，$Q$来自decoder的当前token，$KV$来自encoder的全部输出。因此$KV$通常不变，只需生成一次并缓存。

- 对于`Self Attention`，$QKV$都来自decoder的当前token，因为decoder需要看过去所有的token，因此前面token的$KV$都需要缓存

> 看源码好难——

```python
    def forward(
        self,
        hidden_states: Optional[tuple[torch.FloatTensor]],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> tuple[Union[torch.Tensor, tuple[torch.Tensor]], ...]:
        # 判断是否是Cross Attention
        is_cross_attention = encoder_hidden_states is not None
        # Cross Attention使用cross_attention_cache
        # Self Attention使用self_attention_cache
        # 用is_updated表示当前层的KV是否已缓存 (用于Cross Attention)
        if past_key_value is not None:
            if isinstance(past_key_value, EncoderDecoderCache):
                is_updated = past_key_value.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    curr_past_key_value = past_key_value.cross_attention_cache
                else:
                    curr_past_key_value = past_key_value.self_attention_cache
            else:
                curr_past_key_value = past_key_value


        if is_cross_attention:
            # Cross Attention
            query_states = self.q_attn(hidden_states)
            attention_mask = encoder_attention_mask
            # 尝试获取KV Cache
            if past_key_value is not None and is_updated:
                key_states = curr_past_key_value.layers[self.layer_idx].keys
                value_states = curr_past_key_value.layers[self.layer_idx].values
            else:
                key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
                # 变换成MHA的shape
                shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
                key_states = key_states.view(shape_kv).transpose(1, 2)
                value_states = value_states.view(shape_kv).transpose(1, 2)
        else:
            # Self Attention
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
            shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
            key_states = key_states.view(shape_kv).transpose(1, 2)
            value_states = value_states.view(shape_kv).transpose(1, 2)


        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)

        # 更新缓存: 启动KV Cache，且是Self Attention，或Cross Attention没有缓存过的情况
        if (past_key_value is not None and not is_cross_attention) or (
            past_key_value is not None and is_cross_attention and not is_updated
        ):
            cache_position = cache_position if not is_cross_attention else None
            key_states, value_states = curr_past_key_value.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )
            if is_cross_attention:
                past_key_value.is_updated[self.layer_idx] = True

        # 判断是否是因果注意力 (Casual)
        is_causal = attention_mask is None and query_states.shape[-2] > 1 and not is_cross_attention

        # 选择注意力实现方式
        # [eager/flash_attention_2/sdpa/triton/xformers]
        using_eager = self.config._attn_implementation == "eager"
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # 选择精度提升(upcast)和重排(reorder)
        if using_eager and self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query_states, key_states, value_states, attention_mask, head_mask
            )
        else:
            # 调用注意力计算函数
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                head_mask=head_mask,
                dropout=self.attn_dropout.p if self.training else 0.0,
                is_causal=is_causal,
                **kwargs,
            )

        # 将Attention结果用线性层c_proj投影回原始维度
        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)


        return attn_output, attn_weights
```

同时，KV Cache在减少重复$KV$计算的同时会引入大量的Memory开销，可以粗略计算一下KV Cache的显存占用：

$$
Memory = 2 \times batch\_size \times seq\_len \times num\_layers \times num\_heads \times head\_dims \times dtype\_size
$$

#### MQA

Multi-Query Attention (MQA)是Google在2019年于[《Fast Transformer Decoding: One Write-Head is All You Need》](https://arxiv.org/abs/1911.02150)提出的一种高效注意力机制，旨在减少推理过程中的计算和内存开销。与传统MHA不同，MQA保留多个Query头，但所有注意力头共享同一组Key和Value，这种结构显著减少了KV Cache的Memory开销，同时保持了MHA相近的性能表现。

下面是基于PyTorch的一个简单实现（还没实现Casual Mask）：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 多个Query头
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        # 共享的Key和Value
        self.k_proj = nn.Linear(embed_dim, self.head_dim)
        self.v_proj = nn.Linear(embed_dim, self.head_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        B, T, C = x.size()

        # Q: (B, T, num_heads, head_dim)
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, T, head_dim)

        # K, V: shared (B, T, head_dim)->(B, 1, T, head_dim)
        k = self.k_proj(x).unsqueeze(1)
        v = self.v_proj(x).unsqueeze(1)

        # Attention Score: (B, num_heads, T, T)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Attention output: (B, num_heads, T, head_dim)
        attn_output = torch.matmul(attn_weights, v)

        # 合并头->(B, T, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(attn_output)
```

#### GQA

Grouped-Query Attention (GQA)Google与在023年于[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)提出的一种介于MHA和MQA之间的注意力机制，让多个Query头共享同一组Key和Value，旨在保留部分表达能力的同时大幅减少计算和内存开销。

{{< figure src="img/1.jpg#center" width=600px" title="Overview of grouped-query method">}}

源码上，只在Huggingface的仓库里找到了[sdpa_attention_paged_forward](https://github.com/huggingface/transformers/blob/6dfd561d9cd722dfc09f702355518c6d09b9b4e3/src/transformers/integrations/sdpa_paged.py#L6-L51)的实现，看上去挺GQA的。

核心思路是：

- 先用`repeat_kv`将KV head复制`num_attention_heads // num_key_value_heads`次（从`(B, num_key_value_heads, L, D)`到`(B, num_attention_heads, L, D)`）

- 支持KV Cache的SDPA

## Reference

[大模型推理加速：看图学KV Cache](https://zhuanlan.zhihu.com/p/662498827)

[LM(20)：漫谈 KV Cache 优化方法，深度理解 StreamingLLM](https://zhuanlan.zhihu.com/p/659770503)

[Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)

[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)