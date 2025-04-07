import vdoe

from functools import partial

import math
import numpy as np

import jax
import jax.numpy as jnp

from flax import nnx
import optax

import matplotlib.pyplot as plt

from tqdm import tqdm

SIGMA_LIKELIHOOD = 0.2
DROPOUT_PROB = 0.2
ORDER = 7
POINTS = 16
LATENT_DIM = 32
CONTROL_DIM = 1
TARGET_DIM = 1

class NeuralProcess(vdoe.inference.NeuralProcess):
  def __init__(self, rngs: nnx.Rngs):
    super().__init__(
      encoder=vdoe.nn.AlphaResAttentionBlock(CONTROL_DIM + TARGET_DIM, 32, LATENT_DIM, depth=6, rngs=rngs, activation=nnx.swish),
      decoder=vdoe.nn.AlphaResBlock(LATENT_DIM + CONTROL_DIM, 32, TARGET_DIM, depth=6, rngs=rngs, activation=nnx.swish),
      rngs=rngs
    )

def basis(x: jax.Array, k=ORDER):
  ### Chebyshev basis
  basis_fns = [
    jnp.ones_like(x),
    x
  ]
  for i in range(k - 1):
    a, b = basis_fns[-1], basis_fns[-2]
    basis_fns.append(2 * x * a - b)

  return jnp.stack(basis_fns, axis=-1)

def sample(key: jax.Array, batch, k: int = ORDER):
  return jax.random.normal(key, shape=(batch, k + 1)) / math.sqrt(k + 1)

def evaluate(key: jax.Array, x: jax.Array, w: jax.Array, k: int=ORDER, sigma: float=SIGMA_LIKELIHOOD):
  phi = basis(x, k=k)

  fs = jax.lax.dot_general(
    phi, w,
    dimension_numbers=(
      ([phi.ndim - 1], [w.ndim - 1]),
      (range(phi.ndim - 2), range(w.ndim - 1))
    )
  )
  ys = fs + sigma * jax.random.normal(key, shape=fs.shape)

  return fs, ys

def train(
    seed: int, output: str, epochs: int, batch: int,
    autoencoder: bool=False, subgradient: bool=False,
    log: str | None=None, progress: bool=True
):
  rngs = nnx.Rngs(seed)

  inference = NeuralProcess(rngs=rngs)
  _, state = nnx.split(inference)
  import math
  n_parameters = sum(
    math.prod(x.shape) for x in jax.tree.leaves(state)
  )

  def loss_fn(key, model, xs, ys):
    return vdoe.inference.meta_elbo(
      key, model, xs, ys,
      sigma_noise=SIGMA_LIKELIHOOD,
      autoencoder=autoencoder,
      subgradient=subgradient,
      axes=(1, ),
      l2_reg=1.0e-3,
      sigma_reg=1.0e-3
    )

  @nnx.jit
  def step(key: jax.Array, model, opt):
    key_data, key_loss, key_dropout = jax.random.split(key, num=3)
    xs, _, ys = generate(key_data, batch=batch, points=POINTS, k=ORDER, sigma=SIGMA_LIKELIHOOD)
    loss, grad = nnx.value_and_grad(loss_fn, argnums=1)(key, model, xs[..., None], ys[..., None])
    opt.update(grad)
    return loss

  optimizer = nnx.Optimizer(inference, optax.adabelief(learning_rate=2.0e-4))

  n_iterations = n_parameters // batch
  losses = np.ndarray(shape=(epochs, n_iterations))

  if progress:
    progress_bar = tqdm
  else:
    progress_bar = lambda x, **kwargs: x

  for i in progress_bar(range(epochs), desc='epochs'):
    for j in progress_bar(range(n_iterations), desc='iterations', leave=False):
      key_step = rngs()
      losses[i, j] = step(key_step, inference, optimizer)

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

def generate(key: jax.Array, batch, points, k=ORDER, sigma: float=SIGMA_LIKELIHOOD):
  key_ws, key_xs, key_noise = jax.random.split(key, num=3)

  xs = jnp.sort(
    jax.random.uniform(key_xs, minval=-1, maxval=1, shape=(batch, points)),
    axis=1
  )
  ws = sample(key_ws, batch, k=k)

  fs, ys = evaluate(key_noise, xs, ws, k=k, sigma=sigma)

  return xs, fs, ys

def show(seed: int, log: str, n: int=5, points: int=32, order: int=3):
  rng = jax.random.PRNGKey(seed)

  xs, fs, ys = generate(rng, batch=n * n, points=points, k=order)
  print(xs.shape, ys.shape)

  y_min, y_max = jnp.min(ys), jnp.max(ys)
  y_delta = (y_max - y_min)

  fig = plt.figure(figsize=(4 * n, 4 * n))
  axes = fig.subplots(n, n, squeeze=False).ravel()

  for i in range(n * n):
    axes[i].plot(xs[i], fs[i], color=plt.cm.tab10(0))
    axes[i].scatter(xs[i], ys[i], color=plt.cm.tab10(0))
    axes[i].set_ylim([y_min - 0.05 * y_delta, y_max + 0.05 * y_delta])

  fig.tight_layout()
  fig.savefig(log)
  plt.close(fig)


def doe(seed: int, model: str, log: str, n: int=5, iterations: int=11, progress: bool=True):
  n_grid = 67
  n_test_samples = 163
  rngs = nnx.Rngs(seed)

  inference: NeuralProcess = NeuralProcess(rngs=rngs)
  inference, _ = vdoe.utils.io.load_model(model, inference)

  ws = sample(rngs(), n)

  Xs = np.ndarray(shape=(n, iterations, 1))
  ys = np.ndarray(shape=(n, iterations, 1))
  fs = np.ndarray(shape=(n, iterations, 1))
  Xs_guesses = np.ndarray(shape=(n, iterations, 1))
  fs_guesses = np.ndarray(shape=(n, iterations, 1))

  predictions = np.ndarray(shape=(n, iterations, n_grid, n_test_samples, 1))

  prior_mu_inv_sigma_sqr = np.zeros(shape=(n, LATENT_DIM))
  prior_sum_inv_sigmas_sqr = np.zeros(shape=(n, LATENT_DIM))

  grid = jnp.linspace(-1, 1, num=n_grid)
  eps = jax.random.normal(rngs(), shape=(n, 1, n_test_samples, LATENT_DIM))

  if progress:
    progress_bar = tqdm
  else:
    progress_bar = lambda x: x

  for i in progress_bar(range(iterations)):
    mu, sigma, _ = inference.posterior(prior_mu_inv_sigma_sqr, prior_sum_inv_sigmas_sqr)
    latent = eps * sigma[:, None, None, :] + mu[:, None, None, :]
    grid_ = jnp.broadcast_to(grid[None, :, None, None], shape=(n, n_grid, n_test_samples, 1))
    predictions[:, i] = inference.predict(latent, grid_)

    suggestions = vdoe.inference.suggest(
      rngs(), inference,
      mu_inv_sigma_sqr=prior_mu_inv_sigma_sqr,
      sum_inv_sigmas_sqr=prior_sum_inv_sigmas_sqr,
      sigma_noise=SIGMA_LIKELIHOOD,
      trials=65, n=63
    )

    guesses = vdoe.inference.exploit(
      rngs(), inference,
      mu_inv_sigma_sqr=prior_mu_inv_sigma_sqr,
      sum_inv_sigmas_sqr=prior_sum_inv_sigmas_sqr,
      trials=65, n=63
    )

    f_iter, y_iter = evaluate(rngs(), suggestions[:, None, 0], ws,)
    Xs[:, i, :] = suggestions
    fs[:, i, :] = f_iter[:, 0, None]
    ys[:, i, :] = y_iter[:, 0, None]

    f_guess_iter, y_guess_iter = evaluate(rngs(), guesses[:, None, 0], ws, )
    Xs_guesses[:, i, :] = guesses[:, None, 0]
    fs_guesses[:, i, :] = f_guess_iter[:, None, 0]

    mu_iter, sigma_raw_iter = inference.encoder(jnp.concatenate([suggestions, y_iter], axis=-1)[:, None, :])
    mu_inv_sigma_sqr_iter, sum_inv_sigmas_sqr_iter = inference.aggregate(mu_iter, sigma_raw_iter, axes=())
    prior_mu_inv_sigma_sqr += mu_inv_sigma_sqr_iter[:, 0, :]
    prior_sum_inv_sigmas_sqr += sum_inv_sigmas_sqr_iter[:, 0, :]

  grid = jnp.linspace(-1, 1, num=n_grid)
  f_grid, _ = evaluate(rngs(), jnp.broadcast_to(grid[None, :], shape=(n, n_grid)), ws, )

  fig, axes = plt.subplots(n, iterations, figsize=(3 * iterations, 3 * n), squeeze=False)

  for i in range(n):
    for j in range(iterations):
      low, med, high = np.quantile(predictions[i, j, :, :, 0], q=(0.2, 0.5, 0.8), axis=-1)
      axes[i, j].scatter([Xs_guesses[i, j, 0]], [fs_guesses[i, j, 0]], color=plt.cm.tab10(2), label='guessed' if i == j == 0 else None)
      axes[i, j].scatter([Xs[i, j, 0]], [fs[i, j, 0]], color=plt.cm.tab10(1), label='proposed' if i == j == 0 else None)
      axes[i, j].scatter(Xs[i, :j, 0], ys[i, :j, 0], color=plt.cm.tab10(0), label='observed' if i == j == 0 else None)
      axes[i, j].plot(grid, f_grid[i], color='black', linestyle='--', label='true function' if i == j == 0 else None)
      axes[i, j].fill_between(grid, low, high, alpha=0.25, color=plt.cm.tab10(0), label='20-80 percentiles' if i == j == 0 else None)
      axes[i, j].plot(grid, med, color=plt.cm.tab10(0), label='median' if i == j == 0 else None)

      if i == 0:
        axes[i, j].set_title(f'iteration {j}')

    fig.suptitle(f'Chebyshev Polynomials (order={ORDER})')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncols=6)

  fig.tight_layout(h_pad=1.0)
  fig.savefig(log)
  plt.close(fig)


def test(seed: int, model: str, log: str, n: int=5):
  rngs = nnx.Rngs(seed)

  inference: NeuralProcess = NeuralProcess(rngs=rngs)
  inference, _ = vdoe.utils.io.load_model(model, inference)

  batch = n * n

  xs, fs, ys = generate(rngs(), batch=batch, points=POINTS, k=ORDER, sigma=SIGMA_LIKELIHOOD)
  n_b, n_t = xs.shape

  Xy = np.concatenate([xs[:, :, None], ys[:, :, None]], axis=-1)
  X = xs[:, :, None]

  u = jax.random.uniform(rngs(), shape=(batch, ), minval=0.0, maxval=1.0)
  split = jax.random.uniform(rngs(), shape=(batch, n_t), minval=0.0, maxval=1.0) < u[:, None]

  mus, sigmas_raw = inference.encoder(Xy)

  mu_inv_sigma_sqr, sum_inv_sigmas_sqr = inference.aggregate(mus, sigmas_raw, axes=(1, ), keepdims=True, mask=split)
  print('20-50-80 sigma:', jnp.quantile(1 / jnp.sqrt(jax.nn.softplus(sigmas_raw)), q=jnp.array([0.2, 0.5, 0.8])))

  ### (*, n_e, 1, n_z)
  mu_posterior, sigma_posterior, _ = inference.posterior(mu_inv_sigma_sqr, sum_inv_sigmas_sqr)
  print(mu_posterior.shape, sigma_posterior.shape)

  *_, n_z = mu_posterior.shape
  n_grid = 63
  n_samples = 64
  eps = jax.random.normal(rngs(), shape=(batch, n_grid, n_samples, n_z))

  ### (*, 1, n_samples, n_z)
  latent = eps * sigma_posterior[..., None, :] + mu_posterior[..., None, :]
  print('std latent:', jnp.std(latent))

  ### (*, n_t, n_samples, n_z + n_x)
  grid = jnp.linspace(-1, 1, num=n_grid)
  X_test = jnp.broadcast_to(grid[None, :, None, None], (batch, n_grid, n_samples, 1))
  latent = jnp.broadcast_to(latent, (batch, n_grid, n_samples, n_z))

  ### (*, n_t, n_samples, 1)
  predictions = inference.predict(latent, X_test)

  fig = plt.figure(figsize=(n * 4, n * 4))
  axes = fig.subplots(n, n, squeeze=False).ravel()

  for i in range(batch):
    xs_train = xs[i, split[i] > 0.5]
    ys_train = ys[i, split[i] > 0.5]

    xs_test = xs[i, split[i] < 0.5]
    ys_test = ys[i, split[i] < 0.5]

    axes[i].scatter(xs_train, ys_train, color='black', marker='x')
    axes[i].scatter(xs_test, ys_test, color=plt.cm.tab10(1))

    indx = np.argsort(xs[i])
    axes[i].plot(xs[i, indx], fs[i, indx], linestyle='--', color='black')

    pred = predictions[i, :, :, 0]
    n_q = 2
    qs = np.quantile(pred, q=jnp.linspace(0.0, 1.0, num=2 * n_q + 1), axis=1)
    med = np.median(pred, axis=1)

    for j in range(n_q):
      axes[i].fill_between(grid, qs[j], qs[-j - 1], alpha=1.0 / n_q, color=plt.cm.tab10(0))

    axes[i].plot(grid, med, color=plt.cm.tab10(0))

  min_y = min(np.min(predictions), np.min(ys))
  max_y = max(np.max(predictions), np.max(ys))

  delta = max_y - min_y
  mid = (max_y + min_y) / 2

  for i in range(batch):
      axes[i].set_ylim([mid - 0.525 * delta, mid + 0.525 * delta])

  fig.tight_layout()
  fig.savefig(log)
  plt.close(fig)

if __name__ == '__main__':
  import gearup
  gearup.gearup(
    doe=doe,
    show=show,
    train=train,
    test=test
  )()