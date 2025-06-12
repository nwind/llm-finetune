class LlamaRMSNorm(nn.Module):
  def __init__(self, hidden_size, eps=1e-6):
    super().__init__()
    # 可学习的权重
    self.weight = nn.Parameter(torch.ones(hidden_size))
    # 防止为 0
    self.variance_epsilon = eps

  def forward(self, hidden_states):
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states