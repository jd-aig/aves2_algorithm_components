from PlatformNlp.modules.layer_norm import layer_norm
from PlatformNlp.modules.drop_out import dropout


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
  """Runs layer normalization followed by dropout."""
  output_tensor = layer_norm(input_tensor, name)
  output_tensor = dropout(output_tensor, dropout_prob)
  return output_tensor