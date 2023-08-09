from torch import nn

class HandwrittenDigits(nn.Module):
  def __init__(self, input_shape: int,
               hidden_units: int,
               output_shape: int):
    super().__init__()
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features = input_shape, out_features = hidden_units),
        nn.ReLU(),
        nn.Linear(in_features = hidden_units, out_features = output_shape),
        # nn.ReLU(),
        # nn.Linear(in_features = hidden_units, out_features = output_shape)
    )

  def forward(self,x):
    return self.classifier(x)