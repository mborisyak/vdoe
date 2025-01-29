import vdoe

import numpy as np

import jax
import jax.numpy as jnp

from flax import nnx
import optax

import matplotlib.pyplot as plt

from tqdm import tqdm

SIGMA_LIKELIHOOD = 0.025
LATENT_DIM = 48
HIDDEN_DIM = 64
DROPOUT_PROB = 0.2

class NeuralProcess(vdoe.inference.NeuralProcess):
  def __init__(self, n_input: int, n_output: int, n_latent: int, rngs: nnx.Rngs):
    super().__init__(
      encoder=vdoe.nn.AlphaResAttentionBlock(
        n_input + n_output, HIDDEN_DIM, n_latent, depth=5,
        rngs=rngs, activation=nnx.swish
      ),
      decoder=vdoe.nn.AlphaResBlock(
        n_latent + n_input, HIDDEN_DIM, n_output, depth=5,
        rngs=rngs, activation=nnx.swish
      ),
      rngs=rngs
    )

@jax.jit
def pack(initials, timestamps, values=None):
  n_b, n_exp, n_t, _ = timestamps.shape
  *_, n_init = initials.shape
  initials = jnp.broadcast_to(initials[..., None, :], shape=(n_b, n_exp, n_t, n_init))

  if values is None:
    return jnp.concatenate([initials, timestamps], axis=-1)
  else:
    return jnp.concatenate([initials, timestamps, values], axis=-1)

def train(seed: int, dataset: str, output: str, epochs: int, batch: int, log: str | None=None, progress: bool=True):
  rngs = nnx.Rngs(seed)

  data = np.load(dataset)

  initials, timestamps, samples = data['initial'], data['timestamps'], data['samples']
  total, n_exp, n_init = initials.shape
  *_, n_t = timestamps.shape

  timestamps = jnp.reshape(timestamps, shape=(total, n_exp, n_t, 1))

  print(f'Initials: {initials.shape}')
  print(f'timestamps: {timestamps.shape}')
  print(f'Samples: {samples.shape}')

  inference = NeuralProcess(n_input=1 + n_init, n_output=1, n_latent=LATENT_DIM, rngs=rngs)

  _, state = nnx.split(inference)
  import math
  print(sum(
    math.prod(x.shape) for x in jax.tree.leaves(state)
  ))

  def loss_fn(key, model, init, ts, ys, mask):
    xs = pack(init, ts)

    return vdoe.inference.meta_elbo(
      key, model, xs, ys,
      sigma_noise=SIGMA_LIKELIHOOD,
      autoencoder=True,
      subgradient=True,
      axes=(1, 2),
      mask=mask,
      l2_reg=1.0e-6,
      sigma_reg=1.0e-9
    )

  @nnx.jit
  def step(key: jax.Array, model, init, ts, ys, opt):
    key_loss, key_mask = jax.random.split(key, num=2)
    mask = jax.random.bernoulli(key_mask, p=1 - DROPOUT_PROB)
    loss, grad = nnx.value_and_grad(loss_fn, argnums=1)(key_loss, model, init, ts, ys, mask)
    opt.update(grad)
    return loss

  optimizer = nnx.Optimizer(inference, optax.adabelief(learning_rate=2.0e-4))

  n_iterations = total // batch
  losses = np.ndarray(shape=(epochs, n_iterations))

  if progress:
    progress_bar = tqdm
  else:
    progress_bar = lambda x, **kwargs: x

  for i in progress_bar(range(epochs), desc='epochs'):
    for j in progress_bar(range(n_iterations), desc='iterations', leave=False):
      key_step = rngs()
      key_indx = rngs()
      batch_index = jax.random.randint(key_indx, shape=(batch, ), minval=0, maxval=total)
      losses[i, j] = step(
        key_step, inference,
        initials[batch_index], timestamps[batch_index], samples[batch_index],
        optimizer
      )

  if log is not None:
    fig = plt.figure(figsize=(9, 6))
    axes = fig.subplots()

    low, med, high = np.quantile(losses, q=(0.1, 0.5, 0.9), axis=1)
    mean = np.mean(losses, axis=1)
    axes.plot(np.arange(epochs), med, color=plt.cm.tab10(0), label='median loss')
    axes.plot(np.arange(epochs), mean, color=plt.cm.tab10(0), linestyle='--', label='mean loss')
    axes.fill_between(np.arange(epochs), low, high, color=plt.cm.tab10(0), alpha=0.5, label='10-90 percentiles')
    axes.legend(loc='upper right')
    axes.set_yscale('log')
    fig.tight_layout()
    fig.savefig(log)
    plt.close(fig)

    vdoe.utils.io.save_model(output, inference, aux=dict(losses=losses))

def test(seed: int, dataset: str, model: str, log: str, batch: int, progress: bool=True):
  rngs = nnx.Rngs(seed)

  data = np.load(dataset)
  initials, timestamps, samples = data['initial'], data['timestamps'], data['samples']
  n_total, n_exp, n_init = initials.shape
  *_, n_t = timestamps.shape

  n_train = (2 * n_exp) // 3
  n_test = n_exp - n_train

  timestamps = jnp.reshape(timestamps, shape=(n_total, n_exp, n_t, 1))

  print(f'Initials: {initials.shape}')
  print(f'timestamps: {timestamps.shape}')
  print(f'Samples: {samples.shape}')

  inference = NeuralProcess(n_input=1 + n_init, n_output=1, n_latent=LATENT_DIM, rngs=rngs)
  inference, _ = vdoe.utils.io.load_model(model, inference)

  print(initials[..., :n_train, None, :].shape)
  print(timestamps[..., :n_train, :, :].shape)
  print(samples[..., :n_train, :, :].shape)

  initials = np.broadcast_to(initials[..., None, :], (*samples.shape[:-1], initials.shape[-1]))

  Xy_train = np.concatenate([
    initials[..., :n_train, :, :],
    timestamps[..., :n_train, :, :],
    samples[..., :n_train, :, :]
  ], axis=-1)

  X_test = np.concatenate([
    initials[..., n_train:, :, :],
    timestamps[..., n_train:, :, :],
  ], axis=-1)

  y_test = samples[..., n_train:, :, :]
  assert np.all(np.isfinite(X_test))
  assert np.all(np.isfinite(y_test))

  mus, sigmas_raw = inference.encoder(Xy_train)

  mu_inv_sigma_sqr, sum_inv_sigmas_sqr = inference.aggregate(mus, sigmas_raw, axes=(2, ), keepdims=True, mask=None)
  mu_inv_sigma_sqr = jnp.concatenate([
    jnp.zeros(shape=(n_total, 1, 1, mu_inv_sigma_sqr.shape[-1])),
    jnp.cumsum(mu_inv_sigma_sqr, axis=1)
  ], axis=1)
  sum_inv_sigmas_sqr = jnp.concatenate([
    jnp.ones(shape=(n_total, 1, 1, sum_inv_sigmas_sqr.shape[-1])),
    jnp.cumsum(sum_inv_sigmas_sqr, axis=1)
  ], axis=1)

  print(mu_inv_sigma_sqr.shape)

  print(jnp.quantile(1 / jnp.sqrt(jax.nn.softplus(sigmas_raw)), q=jnp.array([0.2, 0.5, 0.8])))

  ### (*, n_e, 1, n_z)
  mu_posterior, sigma_posterior, _ = inference.posterior(mu_inv_sigma_sqr, sum_inv_sigmas_sqr)
  print(mu_posterior.shape, sigma_posterior.shape)

  *_, n_z = mu_posterior.shape
  n_samples = 64
  eps = jax.random.normal(rngs(), shape=(n_total, n_train + 1, 1, n_samples, n_z))

  ### (*, n_e, 1, n_samples, n_z)
  latent = eps * sigma_posterior[..., None, :] + mu_posterior[..., None, :]
  print(jnp.std(latent))

  ### (*, n_train, n_test, n_t, n_samples, n_z + n_x)
  Xz = jnp.concatenate([
    jnp.broadcast_to(X_test[:, None, :, :, None, :], (n_total, n_train + 1, n_test, n_t, n_samples, X_test.shape[-1])),
    jnp.broadcast_to(latent[:, :, None, :, :, :], (n_total, n_train + 1, n_test, n_t, n_samples, latent.shape[-1])),
  ], axis=-1)

  ### (*, n_train, n_test, n_t, n_samples, 1)
  predictions = inference.decoder(Xz)

  fig = plt.figure(figsize=(n_test * 4 + 4, n_train * 4 + 4))
  axes = fig.subplots(n_train + 1, n_test + 1, squeeze=False)

  display_index = 0

  min_y, max_y = +1e+9, -1e+9

  for j in range(n_train + 1):
    if j > 0:
      ts = timestamps[display_index, j - 1, :, 0]
      ys = samples[display_index, j - 1, :, 0]

      axes[j, 0].scatter(ts, ys, color=plt.cm.tab10(1))

    for i in range(n_test):
      ts = timestamps[display_index, n_train + i, :, 0]
      ys = samples[display_index, n_train + i, :, 0]

      pred = predictions[display_index, j, i, :, :, 0]
      low = np.quantile(pred, q=0.1, axis=1)
      med = np.median(pred, axis=1)
      high = np.quantile(pred, q=0.9, axis=1)

      min_y = min(min_y, np.min(low), np.min(ys))
      max_y = max(max_y, np.max(high), np.max(ys))

      axes[j, i + 1].fill_between(ts, low, high, alpha=0.5, color=plt.cm.tab10(0))
      axes[j, i + 1].plot(ts, med, color=plt.cm.tab10(0))
      axes[j, i + 1].scatter(ts, ys, color=plt.cm.tab10(0))

  delta = max_y - min_y
  mid = (max_y + min_y) / 2

  axes[0, 0].set_title('training samples')
  for i in range(n_test):
    axes[0, i + 1].set_title('test sample')

  for i in range(n_test + 1):
    for j in range(n_train + 1):
      axes[j, i].set_ylim([mid - 0.525 * delta, mid + 0.525 * delta])

  fig.tight_layout()
  fig.savefig(log)
  plt.close(fig)

if __name__ == '__main__':
  import gearup
  gearup.gearup(
    train=train,
    test=test
  )()