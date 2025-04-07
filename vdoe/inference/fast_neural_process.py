from typing import Sequence

import math

import jax
import jax.numpy as jnp

from flax import nnx

from vdoe import utils
from vdoe.inference.common import Inference

__all__ = [
  'FastNeuralProcess',
  'meta_elbo',
  'fast_meta_elbo',
  'expected_entropy'
]

LOG_2_PI = math.log(2 * math.pi)

class FastNeuralProcess(Inference, nnx.Module):
  ### axis for concatenation
  feature_axis: int = 2

  ### axis for adding samples
  set_axis: int = 1

  ### axis for aggregation
  aggregation_axes = (1, )

  def __init__(
    self, encoder: nnx.Module | Sequence[nnx.Module], decoder: nnx.Module, *, rngs: nnx.Rngs
  ):
    if isinstance(encoder, nnx.Module):
      self.encoder = (encoder, )
    else:
      self.encoder = encoder

    self.decoder = decoder

  @classmethod
  def aggregate(cls, mus, sigmas_raw, keepdims: bool = False, mask: jax.Array | None = None):
    inv_sigmas_sqr = jax.nn.softplus(sigmas_raw)

    if mask is None:
      mu_inv_sigma_sqr = jnp.sum(mus * inv_sigmas_sqr, axis=cls.aggregation_axes, keepdims=keepdims)
      sum_inv_sigmas_sqr = jnp.sum(inv_sigmas_sqr, axis=cls.aggregation_axes, keepdims=keepdims)
    else:
      mask = broadcast_left(mask, mus)
      mu_inv_sigma_sqr = jnp.sum(mask * mus * inv_sigmas_sqr, axis=cls.aggregation_axes, keepdims=keepdims)
      sum_inv_sigmas_sqr = jnp.sum(mask * inv_sigmas_sqr, axis=cls.aggregation_axes, keepdims=keepdims)

    return mu_inv_sigma_sqr, sum_inv_sigmas_sqr

  @classmethod
  def posterior(cls, mu_inv_sigma_sqr, sum_inv_sigmas_sqr):
    inv_sigma_sqr_posterior = sum_inv_sigmas_sqr + 1
    mu_posterior = mu_inv_sigma_sqr / inv_sigma_sqr_posterior
    sigma_posterior = jnp.sqrt(1 / inv_sigma_sqr_posterior)

    return mu_posterior, sigma_posterior, inv_sigma_sqr_posterior

  @classmethod
  def concat_context(self, x: jax.Array, y: jax.Array):
    return jnp.concatenate([x, y], axis=self.feature_axis)

  @classmethod
  def append(cls, x_y_1: jax.Array, x_y_2: jax.Array):
    return jnp.concatenate([x_y_1, x_y_2], axis=cls.set_axis)

  @classmethod
  def concat_predictive(self, x: jax.Array, latent: jax.Array):
    aggregation_axes_normalized = [(axis + x.ndim) % x.ndim for axis in self.aggregation_axes]
    latent_shape = tuple(x.shape[i] if i in aggregation_axes_normalized else latent.shape[i] for i in range(x.ndim))

    latent = jnp.broadcast_to(latent, shape=latent_shape)
    return jnp.concatenate([x, latent], axis=self.feature_axis)

  def encode(self, x_y: jax.Array, mask: jax.Array | None=None):
    hidden = x_y

    *intermediate_module, final_module = self.encoder
    for module in intermediate_module:
      mus, sigmas_raw = module(hidden)
      hidden = self.covariant(mus, sigmas_raw, mask=mask)

    return final_module(hidden)

  def infer(self, Xy: jax.Array, keepdims: bool=False, mask: jax.Array | None=None):
    mus, sigmas_raw = self.encode(Xy, mask=mask)

    return self.posterior(
      *self.aggregate(mus, sigmas_raw, keepdims=keepdims, mask=mask)
    )

  def predict(self, X_latent: jax.Array):
    return self.decoder(X_latent)

def masked_sum(xs: jax.Array, mask: jax.Array | None=None, **kwargs):
  if mask is None:
    return jnp.sum(xs, **kwargs)
  else:
    mask_ = broadcast_left(mask, xs)
    return jnp.sum(xs * mask_, **kwargs)

def meta_elbo(
  key: jax.Array, model: NeuralProcess,
  xs: jax.Array, ys: jax.Array,
  sigma_noise: float,
  split: jax.Array,
  mask: jax.Array | None=None,
  autoencoder: bool=False,
  subgradient: bool=False,
  l2_reg: float | None=None,
  sigma_l2_reg: float | None=None,
):
  """
  This function is agnostic to the Neural Process implementation. However, if Neural Process does not aggregate values
  internally, i.e., NP({x_i}) = g(aggregate({f(x_i)}_i)), `fast_meta_elbo` is approx. 3 times faster as it requires
  only a single pass through data.

  Computes ELBO for y_test given x_train, y_train, y_test as in [1]:
      loss(y_test | x_test, x_train, y_train) = -E_{z in q(z | x, y) } [log p(y_test | x_test, z)] +
            KL{q(z | x, y) || q(z | x_train, y_train)}
  where: x = x_train U x_test and y = y_train U y_test.

  Optionally, one can add the autoencoder objective:
      loss(y | x) = -E_{z in q(z | x, y) } [log p(y | x, z)] + KL{q(z | x, y) || P(z)}
  for (x_train, y_train), (x_test, y_test) and combined dataset (x, y).
  This requires only one additional pass through the model (only decoder).

  Optionally, one can disable gradient flow through pseudo-priors q(z | x_train, y_train) and q(z | x_test, y_test)
  bringing the objective closer to the auto-encoder one.

  [1]: Garnelo, M., Schwarz, J., Rosenbaum, D., Viola, F., Rezende, D.J., Eslami, S.M. and Teh, Y.W., 2018.
    Neural processes. arXiv preprint arXiv:1807.01622.
  [2]: Volpp, M., Flürenbrock, F., Grossberger, L., Daniel, C. and Neumann, G., 2021.
    Bayesian Context Aggregation for Neural Processes. In ICLR.

  :param key: random key;
  :param model: a Neural Process;
  :param xs: inputs, an array of shape (batch, set dimension, *features);
  :param ys: target, an array of shape (batch, set dimension, *features);
  :param sigma_noise: std of target noise;
  :param split: a binary mask representing training/test split, must be of shape (batch, *set dimensions);
  :param mask: a binary mask for implementing dropout;
  :param autoencoder: when True, adds auto-encoder loss;
  :param subgradient: when True, disable gradient through pseudo-priors;
  :param l2_reg: scaling for l2 regularization;
  :param sigma_l2_reg: scaling for l2 regularization of `inv_sigma_sqr_raw`.
  :return: combined loss.
  """
  n_batch, *ns_set = split.shape
  feature_ndim = xs.ndim - split.ndim

  key_full, key_train, key_test = jax.random.split(key, num=3)

  mask_train = split
  mask_test = jnp.logical_not(split)
  if mask is not None:
    mask_train = jnp.logical_and(split, mask)
    mask_test = jnp.logical_and(jnp.logical_not(split), mask)

  xs_ys = model.concat_context(xs, ys)

  mus_train, sigmas_raw_train = model.encode(xs_ys, mask=mask_train)
  mus_test, sigmas_raw_test = model.encode(xs_ys, mask=mask_test)
  mus_full, sigmas_raw_full = model.encode(xs_ys, mask=mask)

  mu_inv_sigma_sqr_train, sum_inv_sigmas_sqr_train = model.aggregate(
    mus_train, sigmas_raw_train, keepdims=True, mask=mask_train
  )
  mu_inv_sigma_sqr_test, sum_inv_sigmas_sqr_test = model.aggregate(
    mus_test, sigmas_raw_test, keepdims=True, mask=mask_test
  )
  mu_inv_sigma_sqr_full, sum_inv_sigmas_sqr_full = model.aggregate(
    mus_full, sigmas_raw_full, keepdims=True, mask=mask
  )

  mu_train, sigma_train, inv_sigma_sqr_train = model.posterior(
    mu_inv_sigma_sqr_train, sum_inv_sigmas_sqr_train
  )
  mu_test, sigma_test, inv_sigma_sqr_test = model.posterior(
    mu_inv_sigma_sqr_test, sum_inv_sigmas_sqr_test
  )
  mu_full, sigma_full, inv_sigma_sqr_full = model.posterior(
    mu_inv_sigma_sqr_full, sum_inv_sigmas_sqr_full
  )

  latent_full = jax.random.normal(key_full, shape=mu_full.shape) * sigma_full + mu_full
  xs_latent_full = model.concat_predictive(xs, latent_full)
  predictions_full = model.predict(xs_latent_full)

  ### in a symmetric loss both train and test samples are included
  loss_reco_full = masked_sum(jnp.square((predictions_full - ys) / sigma_noise), mask=mask)

  if subgradient:
    no_grad = jax.lax.stop_gradient
  else:
    no_grad = lambda x: x

  ### sum(sigma_sqr_full / sigma_train) + square(dmu) / sigma_sqr_train + 2 * sum log_sigma_train - 2 * sum log sigma_full
  const = math.prod(latent_full.shape[-feature_ndim:]) * n_batch
  penalty_train = jnp.sum(
    no_grad(inv_sigma_sqr_train) / inv_sigma_sqr_full + jnp.square(mu_train - mu_full) * no_grad(inv_sigma_sqr_train)
    - no_grad(jnp.log1p(sum_inv_sigmas_sqr_train)) + jnp.log1p(sum_inv_sigmas_sqr_full)
  ) - const

  penalty_test = jnp.sum(
    no_grad(inv_sigma_sqr_test) / inv_sigma_sqr_full + jnp.square(mu_test - mu_full) * no_grad(inv_sigma_sqr_test)
    - no_grad(jnp.log1p(sum_inv_sigmas_sqr_test)) + jnp.log1p(sum_inv_sigmas_sqr_full)
  ) - const

  combined_loss = loss_reco_full + penalty_train + penalty_test

  if autoencoder:
    latent_train = jax.random.normal(key_train, shape=mu_train.shape) * sigma_train + mu_train
    latent_test = jax.random.normal(key_test, shape=mu_test.shape) * sigma_test + mu_test

    ### combined latent tensor for computing predictions in one call
    latent_train_test = broadcast_left(mask_train, latent_train) * latent_train + \
                        broadcast_left(mask_test, latent_test) * latent_test

    xs_latent_train_test = model.concat_predictive(xs, latent_train_test)
    predictions_train_test = model.predict(xs_latent_train_test)
    loss_reco_train_test = masked_sum(jnp.square((predictions_train_test - ys) / sigma_noise), mask=mask)

    penalty_ae_train = jnp.sum(1 / inv_sigma_sqr_train + jnp.square(mu_train) + jnp.log1p(sum_inv_sigmas_sqr_train))
    penalty_ae_test = jnp.sum(1 / inv_sigma_sqr_test + jnp.square(mu_test) + jnp.log1p(sum_inv_sigmas_sqr_test))

    penalty_ae_full = jnp.sum(1 / inv_sigma_sqr_full + jnp.square(mu_full) + jnp.log1p(sum_inv_sigmas_sqr_full))

    combined_loss = combined_loss + \
                    loss_reco_train_test + penalty_ae_train + penalty_ae_test + \
                    loss_reco_full + penalty_ae_full

  if l2_reg is not None:
    _, model_state = nnx.split(model)
    regularization = sum(jnp.sum(jnp.square(x)) for x in jax.tree.leaves(model_state))

    combined_loss = combined_loss + l2_reg * regularization

  if sigma_l2_reg is not None:
    combined_loss = combined_loss + sigma_l2_reg * (
      masked_sum(jnp.square(sigmas_raw_test), mask=mask_test) +
      masked_sum(jnp.square(sigmas_raw_train), mask=mask_train)
    )

  return combined_loss / (n_batch * math.prod(ns_set))


def fast_meta_elbo(
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
  **This function is only applicable if encoder does not aggregate internally, i.e., consists of only one module.**

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
  key: jax.Array, model: NeuralProcess, x_context: jax.Array, y_context: jax.Array, x: jax.Array,
  sigma_noise: float, n: int, mask_context: jax.Array | None=None
):
  """
  Computes estimate of the expected entropy of the posterior after observing (x, y)
  where y is drawn from the current prior.

  :param key: PRNG key;
  :param model: neural process;
  :param x: controls;
  :param x_context: controls defining the prior;
  :param y_context: labels defining the prior;
  :param sigma_noise: std of the observational noise;
  :param n: n latent parameters + a set of observations is drawn from the prior;
  :param mask_context: mask for the context values;
  :return: estimate of the expected posterior entropy.
  """
  xy_context = model.concat_context(x_context, y_context)
  mu, sigma, _ = model.infer(xy_context, keepdims=True, mask=mask_context)
  n_batch, *feature_shape_latent = mu.shape
  _, *feature_shape_control = x.shape

  key_latent, key_noise = jax.random.split(key, num=2)

  latent = jax.random.normal(key_latent, shape=(n_batch, n, *feature_shape_latent)) * sigma[:, None, ...] + mu[:, None, ...]
  x = jnp.broadcast_to(x[:, None, ...], shape=(n_batch, n, *feature_shape_control))

  latent_flat = jnp.reshape(latent, shape=(n_batch * n, *feature_shape_latent))
  x_flat = jnp.reshape(x, shape=(n_batch * n, *feature_shape_control))

  ### (n_batch * n, *set, ...)
  x_latent_flat = model.concat_predictive(x_flat, latent_flat)
  y_flat = model.predict(x_latent_flat)
  y_flat_noisy = jax.random.normal(key_noise, shape=y_flat.shape) * sigma_noise + y_flat
  xy_virtual = model.concat_context(x_flat, y_flat_noisy)

  xy_context_br = jnp.broadcast_to(xy_context[:, None, ...], shape=(n_batch, n, *xy_context.shape[1:]))
  xy_context_flat = jnp.reshape(xy_context_br, shape=(n_batch * n, *xy_context.shape[1:]))
  xy_combined = model.append(xy_context_flat, xy_virtual)

  if mask_context is None:
    mask_combined = None
  else:
    _, *mask_shape = mask_context.shape
    mask_context_br = jnp.broadcast_to(mask_context[:, None, ...], shape=(n_batch, n, *mask_shape))
    mask_context_flat = jnp.reshape(mask_context_br, shape=(n_batch * n, *mask_shape))

    set_axis_normalized = (model.set_axis + xy_context.ndim) % xy_context.ndim
    mask_virtual_shape = tuple(
      xy_virtual.shape[i] if i == set_axis_normalized else mask_context_flat.shape[i]
      for i in range(mask_context_flat.ndim)
    )
    mask_combined = jnp.concatenate([
      mask_context_flat,
      jnp.ones(shape=mask_virtual_shape, dtype=mask_context_flat.dtype)
    ], axis=set_axis_normalized)

  mus, sigmas_raw = model.encode(xy_combined, mask=mask_combined)
  mu_inv_sigma_sqr_combined, sum_inv_sigmas_combined = model.aggregate(mus, sigmas_raw)
  log_sigma_sqr_combined = -jnp.log1p(sum_inv_sigmas_combined)
  _, n_z = log_sigma_sqr_combined.shape

  log_sigma_sqr_combined = jnp.reshape(log_sigma_sqr_combined, shape=(n_batch, n, n_z))

  expected_entropy = 0.5 * (n_z * (1 + LOG_2_PI) + jnp.sum(log_sigma_sqr_combined, axis=-1))

  return expected_entropy

def fast_expected_entropy(
  key: jax.Array, model: NeuralProcess, x: jax.Array,
  prior_mu_inv_sigma_sqr: jax.Array, prior_sum_inv_sigmas_sqr: jax.Array,
  sigma_noise: float,
  n: int
):
  """
  **This function is only applicable if encoder does not aggregate internally, i.e., consists of only one module.**

  Computes estimate of the expected entropy of the posterior after observing (x, y)
  where y is drawn from the current prior.

  :param key: PRNG key;
  :param model: neural process;
  :param x: controls;
  :param prior_mu_inv_sigma_sqr: mu / sigma^2 where mu and sigma^2 are the mean and variance of the prior;
  :param prior_sum_inv_sigmas_sqr: 1 / sigma^2 where mu and sigma^2 are the mean and variance of the prior;
  :param sigma_noise: std of the observational noise;
  :param n: n latent parameters + a set of observations is drawn from the prior;
  :return: estimate of the expected posterior entropy.
  """
  mu, sigma, _ = model.posterior(prior_mu_inv_sigma_sqr, prior_sum_inv_sigmas_sqr)
  *batch, n_z = mu.shape
  *_, n_x = x.shape

  key_latent, key_noise = jax.random.split(key, num=2)

  latent = jax.random.normal(key_latent, shape=(*batch, n, n_z)) * sigma[..., None, :] + mu[..., None, :]
  x = jnp.broadcast_to(x[..., None, :], shape=(*batch, n, n_x))
  y = model.predict(latent, x)
  y_noisy = jax.random.normal(key_noise, shape=y.shape) * sigma_noise + y

  sample_mu, sample_sigma_raw = model.encoder(jnp.concatenate([x, y_noisy], axis=model.feature_axis))
  _, sample_sum_inv_sigmas_sqr = model.aggregate(sample_mu, sample_sigma_raw, axes=())
  expected_sum_inv_sigmas_sqr = prior_sum_inv_sigmas_sqr[..., None, :] + sample_sum_inv_sigmas_sqr

  expected_log_sigmas = -0.5 * jnp.log1p(expected_sum_inv_sigmas_sqr)
  expected_entropy = jnp.sum(expected_log_sigmas, axis=-1)
  return jnp.mean(expected_entropy, axis=-1)