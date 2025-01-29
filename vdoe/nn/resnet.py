import jax
import jax.numpy as jnp

from flax import nnx

__all__ = [
  'AlphaResBlock',
  'AlphaResAttentionBlock'
]

class AlphaResBase(nnx.Module):
  def __init__(self, n_in: int, n_hidden: int, depth: int, *, rngs: nnx.Rngs, activation=nnx.leaky_relu):
    self.embedding = nnx.Linear(n_in, n_hidden, rngs=rngs)

    self.hidden = list()
    self.alphas = list()

    for i in range(depth):
      self.hidden.append(
        nnx.LinearGeneral(n_hidden, n_hidden, rngs=rngs)
      )
      self.alphas.append(
        nnx.Param(jnp.zeros(shape=()), )
      )

    self.activation = activation

  def __call__(self, X: jax.Array):
    result = self.activation(self.embedding(X))

    for layer, alpha in zip(self.hidden, self.alphas):
      hidden = self.activation(layer(result))
      result = result + alpha * hidden

    return result

class AlphaResBlock(AlphaResBase):
  def __init__(self, n_in: int, n_hidden: int, n_out: int, depth: int, *, rngs: nnx.Rngs, activation=nnx.leaky_relu):
    super().__init__(n_in, n_hidden, depth=depth, rngs=rngs, activation=activation)
    self.output = nnx.Linear(n_hidden, n_out, rngs=rngs)

  def __call__(self, X: jax.Array):
    result = super().__call__(X)
    return self.output(result)

class AlphaResAttentionBlock(AlphaResBase):
  def __init__(self, n_in: int, n_hidden: int, n_out: int, depth: int, *, rngs: nnx.Rngs, activation=nnx.leaky_relu):
    super().__init__(n_in, n_hidden, depth=depth, rngs=rngs, activation=activation)

    self.output_weights = nnx.Linear(n_hidden, n_out, rngs=rngs)
    self.output_values = nnx.Linear(n_hidden, n_out, rngs=rngs)

  def __call__(self, X: jax.Array):
    result = super().__call__(X)
    return self.output_values(result), self.output_weights(result)
