from typing import Sequence

import jax

from flax import nnx

__all__ = [
  'MLPBlock',
  'MLPAttentionBlock'
]

class MLPBlock(nnx.Module):
  def __init__(self, units: Sequence[int], *, rngs: nnx.Rngs, activation=nnx.leaky_relu):
    self.layers = [
      nnx.Linear(n_in, n_out, rngs=rngs)
      for n_in, n_out in zip(units[:-1], units[1:])
    ]

    self.activation = activation

  def __call__(self, X: jax.Array):
    result = X

    *hidden, output = self.layers

    for layer in hidden:
      result = self.activation(layer(result))

    return output(result)

class MLPAttentionBlock(nnx.Module):
  def __init__(self, units: Sequence[int], *, rngs: nnx.Rngs, activation=nnx.leaky_relu):
    self.hidden = [
      nnx.Linear(n_in, n_out, rngs=rngs)
      for n_in, n_out in zip(units[:-2], units[1:-1])
    ]

    *_, n_last_hidden, n_output = units

    self.output = (
      nnx.Linear(n_last_hidden, n_output, rngs=rngs),
      nnx.Linear(n_last_hidden, n_output, rngs=rngs)
    )

    self.activation = activation

  def __call__(self, X: jax.Array):
    result = X

    for layer in self.hidden:
      result = self.activation(layer(result))

    out1, out2 = self.output
    return out1(result), out2(result)