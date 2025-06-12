class LlamaDecoderLayer(nn.Module):
  def __init__(self, config: LlamaConfig, layer_idx: int):
    super().__init__()
    self.hidden_size = config.hidden_size

    # attention 计算，就是前面介绍过的 QKV 计算
    self.self_attn = LlamaAttention(config=config,
                                    layer_idx=layer_idx)

    # MLP 层，也叫 FFA，后面会详细介绍
    self.mlp = LlamaMLP(config)
    self.input_layernorm = LlamaRMSNorm(config.hidden_size)
    self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size)

  def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.LongTensor],
    **kwargs,
  ):
    residual = hidden_states

    # 输入归一化
    hidden_states = self.input_layernorm(hidden_states)

    # 计算注意力
    hidden_states = self.self_attn(
      hidden_states=hidden_states,
      attention_mask=attention_mask,
      position_ids=position_ids,
      **kwargs,
    )
    hidden_states = residual + hidden_states

    # 输出归一化
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    # 残差连接
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)
    return outputs