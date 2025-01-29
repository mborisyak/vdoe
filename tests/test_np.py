import jax
import jax.numpy as jnp
from flax import nnx
import vdoe


def test_neural_process(seed):
  rng = jax.random.PRNGKey(seed)
  rng, key_rngs = jax.random.split(rng, num=2)

  rngs = nnx.Rngs(key_rngs)

  encoder = vdoe.nn.MLPAttentionBlock([3, 4, 5], rngs=rngs)
  decoder = vdoe.nn.MLPBlock([5 + 2, 3, 13], rngs=rngs)

  inference = vdoe.inference.NeuralProcess(encoder=encoder, decoder=decoder, rngs=rngs)

  rng, key_X_train, key_X_test, key_y_train, key_mask, key_pred = jax.random.split(rng, num=6)
  X_train = jax.random.normal(key=key_X_train, shape=(4, 11, 17, 2))
  X_test = jax.random.normal(key=key_X_test, shape=(4, 12, 18, 2))
  y_train = jax.random.normal(key=key_y_train, shape=(4, 11, 17, 1))
  mask = jax.random.bernoulli(key=key_X_train, shape=(4, 11, 17))

  mu, sigma, _ = inference.infer(X_train, y_train, axes=(1, 2))
  assert mu.shape == (4, 5)
  assert sigma.shape == (4, 5)

  mu, sigma, _ = inference.infer(X_train, y_train, axes=(1, 2), keepdims=True)
  assert mu.shape == (4, 1, 1, 5)
  assert sigma.shape == (4, 1, 1, 5)

  mu, sigma, _ = inference.infer(X_train, y_train, axes=(1, 2), mask=mask)
  assert mu.shape == (4, 5)
  assert sigma.shape == (4, 5)

  mu, sigma, _ = inference.infer(X_train, y_train, axes=(1, 2), keepdims=True, mask=mask)
  assert mu.shape == (4, 1, 1, 5)
  assert sigma.shape == (4, 1, 1, 5)

  latent = jnp.broadcast_to(mu + sigma, (*X_test.shape[:-1], mu.shape[-1]))
  y_pred = inference.predict(latent, X_test)
  assert y_pred.shape == (4, 12, 18, 13)

  y_pred = inference(key_pred, X_train, y_train, X_test, axes=(1, 2), mask=mask)
  assert y_pred.shape == (4, 12, 18, 13)

def test_blocks(seed):
  rng = jax.random.PRNGKey(seed)
  rng, key_rngs = jax.random.split(rng, num=2)

  rngs = nnx.Rngs(key_rngs)

  encoder = vdoe.nn.AlphaResAttentionBlock(3, 48, 32, depth=3, rngs=rngs, activation=nnx.swish)

  rng, key_X_train, key_X_test, key_y_train, key_mask, key_pred = jax.random.split(rng, num=6)
  X_train = jax.random.uniform(key=key_X_train, shape=(4, 11, 17, 2))
  X_test = jax.random.normal(key=key_X_test, shape=(4, 12, 18, 2))
  y_train = jax.random.uniform(key=key_y_train, shape=(4, 11, 17, 1))
  mask = jax.random.bernoulli(key=key_X_train, shape=(4, 11, 17))

  Xy_train = jnp.concatenate([X_train, y_train], axis=-1)

  mus, sigmas_raw = encoder(Xy_train)

  print(jnp.std(mus))
  print(jnp.std(sigmas_raw))