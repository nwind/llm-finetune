class LlamaMLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    # 隐藏层维度 4096
    hidden_size = config.hidden_size
    # 中间状态维度 11008
    intermediate_size = config.intermediate_size

    self.gate_proj = nn.Linear(hidden_size,
                               intermediate_size)
    self.up_proj = nn.Linear(hidden_size,
                             intermediate_size)
    self.down_proj = nn.Linear(intermediate_size,
                               hidden_size)

  def forward(self, x):
    down_proj = self.down_proj(
                    self.silu(self.gate_proj(x))
                    * self.up_proj(x)
                )
    return down_proj