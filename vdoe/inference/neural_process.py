from typing import Sequence

import math

import jax
import jax.numpy as jnp

from flax import nnx

__all__ = [
  'NeuralProcess',
  'meta_elbo',
  'suggest'
]

def broadcast_left(array, target):
  if isinstance(target, jax.Array):
    target_ndim = target.ndim
  elif isinstance(target, tuple):
    target_ndim = len(target)
  elif isinstance(target, int):
    target_ndim = target
  else:
    raise ValueError()

  broadcast = (
    *(slice(None, None, None) for _ in range(array.ndim)),
    *(None for _ in range(target_ndim - array.ndim)),
  )

  return array[broadcast]

class NeuralProcess(nnx.Module):
  def __init__(self, encoder: nnx.Module, decoder: nnx.Module, *, rngs: nnx.Rngs):
    self.encoder = encoder
    self.decoder = decoder

  @staticmethod
  def aggregate(mus, sigmas_raw, axes: int | Sequence[int], keepdims: bool = False, mask: jax.Array | None = None):
    inv_sigmas_sqr = jax.nn.softplus(sigmas_raw)

    if mask is None:
      mu_inv_sigma_sqr = jnp.sum(mus * inv_sigmas_sqr, axis=axes, keepdims=keepdims)
      sum_inv_sigmas_sqr = jnp.sum(inv_sigmas_sqr, axis=axes, keepdims=keepdims)
    else:
      mask = broadcast_left(mask, mus)
      mu_inv_sigma_sqr = jnp.sum(mask * mus * inv_sigmas_sqr, axis=axes, keepdims=keepdims)
      sum_inv_sigmas_sqr = jnp.sum(mask * inv_sigmas_sqr, axis=axes, keepdims=keepdims)

    return mu_inv_sigma_sqr, sum_inv_sigmas_sqr

  @staticmethod
  def posterior(mu_inv_sigma_sqr, sum_inv_sigmas_sqr):
    inv_sigma_sqr_posterior = sum_inv_sigmas_sqr + 1
    mu_posterior = mu_inv_sigma_sqr / inv_sigma_sqr_posterior
    sigma_posterior = jnp.square(1 / inv_sigma_sqr_posterior)

    return mu_posterior, sigma_posterior, inv_sigma_sqr_posterior

  def infer(
    self, X: jax.Array, y: jax.Array,
    axes: int | Sequence[int], keepdims: bool=False,
    mask: jax.Array | None=None,
  ):
    ### (batch, experiments, time, features)
    Xy = jnp.concatenate([X, y], axis=-1)
    mus, sigmas_raw = self.encoder(Xy)
    return self.posterior(
      *self.aggregate(mus, sigmas_raw, axes=axes, keepdims=keepdims, mask=mask)
    )

  def predict(self, latent: jax.Array, X: jax.Array):
    latent = jnp.broadcast_to(latent, shape=(*X.shape[:-1], latent.shape[-1]))
    X_latent = jnp.concatenate([X, latent], axis=-1)
    return self.decoder(X_latent)

  def __call__(
    self, key: jax.Array,
    X_context: jax.Array, y_context: jax.Array, X: jax.Array,
    axes: int | Sequence[int],
    mask: jax.Array | None=None,
  ):
    mus, sigmas, _, _ = self.infer(X_context, y_context, axes=axes, keepdims=True, mask=mask)
    latent = jax.random.normal(key, shape=mus.shape) * sigmas + mus
    return self.predict(latent, X)

def meta_elbo(
  key: jax.Array, model: NeuralProcess,
  xs: jax.Array, ys: jax.Array,
  sigma_noise: float,
  axes: int | Sequence[int]=(2, ),
  mask: jax.Array | None=None,
  autoencoder: bool=False,
  subgradient: bool=False,
  l2_reg: float | None=None,
  sigma_reg: float | None=None,
):
  """
  Computes ELBO for y_test given x_train, y_train, y_test as in [1]:
      loss(y_test | x_test, x_train, y_train) = -E_{z in q(z | x, y) } [log p(y_test | x_test, z)] +
            KL{q(z | x, y) || q(z | x_train, y_train)}
  where: x = x_train U x_test and y = y_train U y_test.

  The function assumes `model` follows Bayesian Context Aggregation [2].
      mu, inv_sigma_sqr_raw = model(Xy)
      1 / sigma ** 2 = softplus(inv_sigma_sqr_raw)

  The function randomly splits X and y into two sets and computes symmetric loss:
      loss(y_1 | x_1, x_2, y_2) + loss(y_2 | x_2, x_1, y_1)

  The splits are uniform in the number of elements:
      p ~ U[0, 1]
      train ~ Bernoulli(p)
      test = not train
  NB: although the function supports arbitrary dimensions (3+) of inputs,
    train/test splits allways follows the second axis (axis=1).

  Optionally, one can add the autoencoder objective:
      loss(y | x) = -E_{z in q(z | x, y) } [log p(y | x, z)] + KL{q(z | x, y) || P(z)}
  for (x_train, y_train), (x_test, y_test) and (x, y). This requires an additional pass through decoder.

  Optionally, one can disable gradient flow through pseudo-priors q(z | x_train, y_train) and q(z | x_test, y_test)
  bringing the objective closer to the auto-encoder one.

  Additionally, a uniform sample dropout is applied when `dropout_prob` is not None.

  [1]: Garnelo, M., Schwarz, J., Rosenbaum, D., Viola, F., Rezende, D.J., Eslami, S.M. and Teh, Y.W., 2018.
    Neural processes. arXiv preprint arXiv:1807.01622.
  [2]: Volpp, M., Flürenbrock, F., Grossberger, L., Daniel, C. and Neumann, G., 2021.
    Bayesian Context Aggregation for Neural Processes. In ICLR.

  :param key: random key;
  :param model: a Neural Process;
  :param xs: inputs, an array of shape (batch, set, *, input features);
  :param ys: target, an array of shape (batch, set, *, target features);
  :param sigma_noise: std of target noise;
  :param axes: reduction axes;
  :param mask: a mask to be applied, must be of shape (batch, set, *),
    might be useful for implementing time/spatial dropout, ignored if None.
  :param autoencoder: when True, adds auto-encoder loss;
  :param subgradient: when True, disable gradient through pseudo-priors;
  :param l2_reg: scaling for l2 regularization;
  :param sigma_reg: scaling for l2 regularization of `inv_sigma_sqr_raw`.
  :return: loss.
  """
  ### computes loss for test given train and train given test
  ### splits are random
  n_b, n_exp, *n_t, _ = ys.shape
  key_split, key_u, key_dropout, key_latent = jax.random.split(key, num=4)

  ### compound distribution ensures uniform distribution
  ### of the number of accepted experiments
  u = jax.random.uniform(key_u, shape=(n_b, ))
  split = jax.random.uniform(key_split, shape=(n_b, n_exp)) < u[:, None]
  mask_train = split
  mask_test = jnp.logical_not(split)

  if mask is not None:
    mask_train = jnp.logical_and(broadcast_left(mask_train, mask), mask)
    mask_test = jnp.logical_and(broadcast_left(mask_test, mask), mask)

  X = xs
  Xy = jnp.concatenate([xs, ys], axis=-1)

  mus, sigmas_raw = model.encoder(Xy)

  mu_inv_sigma_sqr_train, sum_inv_sigmas_sqr_train = model.aggregate(
    mus, sigmas_raw, axes=axes, keepdims=True, mask=mask_train
  )

  mu_inv_sigma_sqr_test, sum_inv_sigmas_sqr_test = model.aggregate(
    mus, sigmas_raw, axes=axes, keepdims=True, mask=mask_test
  )

  mu_inv_sigma_sqr_full = mu_inv_sigma_sqr_train + mu_inv_sigma_sqr_test
  sum_inv_sigmas_sqr_full = sum_inv_sigmas_sqr_train + sum_inv_sigmas_sqr_test

  mu_train, sigma_train, inv_sigma_sqr_train = model.posterior(
    mu_inv_sigma_sqr_train, sum_inv_sigmas_sqr_train
  )
  mu_test, sigma_test, inv_sigma_sqr_test = model.posterior(
    mu_inv_sigma_sqr_test, sum_inv_sigmas_sqr_test
  )
  mu_full, sigma_full, inv_sigma_sqr_full = model.posterior(
    mu_inv_sigma_sqr_full, sum_inv_sigmas_sqr_full
  )

  latent_full = jax.random.normal(key_latent, shape=mu_full.shape) * sigma_full + mu_full
  predictions_full = model.predict(latent_full, X)
  if mask is None:
    loss_reco_full = jnp.sum(jnp.square((predictions_full - ys) / sigma_noise))
  else:
    loss_reco_full = jnp.sum(jnp.square((predictions_full - ys) / sigma_noise) * mask)

  if subgradient:
    no_grad = jax.lax.stop_gradient
  else:
    no_grad = lambda x: x

  ### sum(sigma_sqr_full / sigma_train) + square(dmu) / sigma_sqr_train + 2 * sum log_sigma_train - 2 * sum log sigma_full
  const = latent_full.shape[-1] * n_b
  penalty_train = jnp.sum(
    no_grad(inv_sigma_sqr_train) / inv_sigma_sqr_full + jnp.square(mu_train - mu_full) * no_grad(inv_sigma_sqr_train)
    -no_grad(jnp.log1p(sum_inv_sigmas_sqr_train)) + jnp.log1p(sum_inv_sigmas_sqr_full)
  ) - const

  penalty_test = jnp.sum(
    no_grad(inv_sigma_sqr_test) / inv_sigma_sqr_full + jnp.square(mu_test - mu_full) * no_grad(inv_sigma_sqr_test)
    -no_grad(jnp.log1p(sum_inv_sigmas_sqr_test)) + jnp.log1p(sum_inv_sigmas_sqr_full)
  ) - const

  combined_loss = loss_reco_full + penalty_train + penalty_test

  if autoencoder:
    latent_train = jax.random.normal(key_latent, shape=mu_train.shape) * sigma_train + mu_train
    latent_test = jax.random.normal(key_latent, shape=mu_test.shape) * sigma_test + mu_test

    ### combined latent tensor for computing predictions in one call
    latent_train_test = broadcast_left(split, latent_train) * latent_train + \
                        broadcast_left(1 - split, latent_test) * latent_test

    predictions_train_test = model.predict(latent_train_test, X)
    if mask is None:
      loss_reco_train_test = jnp.sum(jnp.square((predictions_train_test - ys) / sigma_noise))
    else:
      loss_reco_train_test = jnp.sum(jnp.square((predictions_train_test - ys) / sigma_noise) * mask)

    penalty_ae_train = jnp.sum(1 / inv_sigma_sqr_train + jnp.square(mu_train) + jnp.log1p(sum_inv_sigmas_sqr_train))
    penalty_ae_test = jnp.sum(1 / inv_sigma_sqr_test + jnp.square(mu_test) + jnp.log1p(sum_inv_sigmas_sqr_test))

    penalty_ae_full = jnp.sum(1 / inv_sigma_sqr_full + jnp.square(mu_full) + jnp.log1p(sum_inv_sigmas_sqr_full))

    combined_loss = combined_loss + \
                    loss_reco_train_test + penalty_ae_train + penalty_ae_test + \
                    loss_reco_full + penalty_ae_full

  if l2_reg is not None:
    _, model_state = nnx.split(model)
    n_parameters = sum(math.prod(x.shape) for x in jax.tree.leaves(model_state))
    regularization = sum(jnp.sum(jnp.square(x)) for x in jax.tree.leaves(model_state)) / n_parameters

    combined_loss = combined_loss + l2_reg * regularization

  if sigma_reg is not None:
    combined_loss = combined_loss + sigma_reg * jnp.mean(jnp.square(sigmas_raw))

  return combined_loss / (n_b * math.prod(n_t))

def expected_entropy(
  key: jax.Array, model: NeuralProcess, x: jax.Array,
  prior_mu_inv_sigma_sqr: jax.Array, prior_sum_inv_sigmas_sqr: jax.Array,
  sigma_noise: float,
  n: int
):
  mu, sigma, _ = model.posterior(prior_mu_inv_sigma_sqr, prior_sum_inv_sigmas_sqr)
  *batch, n_z = mu.shape
  *_, n_x = x.shape

  key_latent, key_noise = jax.random.split(key, num=2)

  latent = jax.random.normal(key_latent, shape=(*batch, n, n_z)) * sigma[..., None, :] + mu[..., None, :]
  x = jnp.broadcast_to(x[..., None, :], shape=(*batch, n, n_x))
  y = model.predict(latent, x)
  y_noisy = jax.random.normal(key_noise, shape=y.shape) * sigma_noise + y

  sample_mu, sample_sigma_raw = model.encoder(jnp.concatenate([x, y_noisy], axis=-1))
  _, sample_sum_inv_sigmas_sqr = model.aggregate(sample_mu, sample_sigma_raw, axes=())
  expected_sum_inv_sigmas_sqr = prior_sum_inv_sigmas_sqr[..., None, :] + sample_sum_inv_sigmas_sqr

  expected_log_sigmas = -0.5 * jnp.log1p(expected_sum_inv_sigmas_sqr)
  expected_entropy = jnp.sum(expected_log_sigmas, axis=-1)
  return jnp.mean(expected_entropy, axis=-1)

def suggest(key, model: NeuralProcess, mu_inv_sigma_sqr: jax.Array, sum_inv_sigmas_sqr: jax.Array, sigma_noise: float, trials: int, n: int):
  key_initial, key_entropy = jax.random.split(key, num=2)
  *batch, n_z = mu_inv_sigma_sqr.shape

  loss_fn = nnx.jit(expected_entropy, static_argnames=('n', ))

  probes = jax.random.uniform(key_initial, shape=(*batch, trials, 1), minval=-1, maxval=1)
  entropy = loss_fn(
    key_entropy, model, probes,
    prior_mu_inv_sigma_sqr=jnp.broadcast_to(mu_inv_sigma_sqr[..., None, :], shape=(*batch, trials, n_z)),
    prior_sum_inv_sigmas_sqr=jnp.broadcast_to(sum_inv_sigmas_sqr[..., None, :], shape=(*batch, trials, n_z)),
    sigma_noise=sigma_noise,
    n=n
  )

  best = jnp.argmin(entropy, axis=-1)

  n = math.prod(batch)
  probes_ = jnp.reshape(probes, shape=(n, trials, 1))

  return probes_[jnp.arange(n), best.ravel(), :].reshape((*batch, 1))