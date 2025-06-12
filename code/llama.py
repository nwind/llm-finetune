class LlamaModel(LlamaPreTrainedModel):
  def __init__(self, config: LlamaConfig):
    super().__init__(config)
    self.padding_idx = config.pad_token_id
    # 词表大小
    self.vocab_size = config.vocab_size

    # 词嵌入
    self.embed_tokens = nn.Embedding(
      config.vocab_size, config.hidden_size, self.padding_idx)

    # 解码层，后面会展开介绍
    self.layers = nn.ModuleList(
      [LlamaDecoderLayer(config, layer_idx)
        for layer_idx in range(config.num_hidden_layers)]
    )

    self.norm = LlamaRMSNorm(config.hidden_size,
                 eps=config.rms_norm_eps)

  def forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    **kwargs,
  ):
    inputs_embeds = self.embed_tokens(input_ids)

    causal_mask = self._update_causal_mask(
      attention_mask, inputs_embeds
    )

    hidden_states = inputs_embeds

    for decoder_layer in self.layers:
      layer_outputs = decoder_layer(
        hidden_states,
        attention_mask=causal_mask,
        position_ids=position_ids
      )

      hidden_states = layer_outputs[0]

    hidden_states = self.norm(hidden_states)

    return hidden_states