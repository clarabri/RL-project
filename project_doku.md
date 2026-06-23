# PPO

**Proximal Policy Optimization** (Schulman et al., 2017) is an on-policy,
actor-critic policy-gradient method. The actor is trained with a *clipped
surrogate objective* that keeps every update close to the policy that collected
the data. That constraint is what lets a single rollout be reused for several
optimisation epochs without the update diverging, which makes PPO far more
sample-efficient than vanilla policy gradients while staying simpler than TRPO.

This implementation targets pixelnbased Atari (Breakout) and follows the same
contract as the other algorithms in this template: every learning-relevant knob
lives on `PPOAlgorithm.__init__`, and `step()` reads like the PPO update.


## Networks

Both heads read the same stacked pixel observation (`obs_key: pixels`) but are
**independent networks —> no shared encoder**. Each is a NaturenCNN
(`ConvNet`, 32/64/64 channels, 8×8 / 4×4 / 3×3 kernels, strides 4 / 2 / 1)
followed by a single 512 unit MLP head:

- **Actor** (`make_atari_ppo_actor`) → one logit per action, wrapped by a
  `ProbabilisticActor` into a `Categorical` policy (`return_log_prob=True`).
- **Critic** (`make_atari_ppo_critic`) → a single scalar `V(s)`, wrapped by a
  `ValueOperator`.

Both factories live in `src/networks.py` and are bound through the algorithm
config as `_partial_` targets, so the conv/MLP shapes can be tuned from YAML
without touching the algorithm.

## Implementation

The algorithm maps onto the `BaseAlgorithm` API:

| Method | What PPO does |
|--------|---------------|
| `setup(make_env)` | Reads the observation/action specs, builds the actor and critic (above), the `ClipPPOLoss` (with `normalize_advantage=True`), the `GAE` module, a `TensorDictReplayBuffer` over a `LazyTensorStorage` with `SamplerWithoutReplacement`, and the Adam optimiser. Also pre computes `_total_updates` for the anneal schedule. |
| `step(batch)` | Flattens the batch, computes GAE under `no_grad`, refills the buffer, then runs `ppo_epochs × minibatches` of (loss → backward → grad-norm clip → optimiser step), annealing `lr` and `clip_epsilon` each update. Returns mean actor/critic/entropy losses. |
| `get_policy()` / `get_explore_policy()` | Both return the stochastic actor — PPO collects *and* evaluates with the same policy. |
| `get_collector_config()` | Reports `frames_per_batch`, `init_random_frames`, `max_frames_per_traj` to the trainer. |

Notes:

- Advantages are computed under `no_grad` **before** the epoch loop, so they stay
  fixed while the policy and value function are updated against the old estimates.
- The optimiser, loss, GAE, and buffer are all built from the single
  `ClipPPOLoss` parameter set, checkpoints save `loss_module.state_dict()`
  (actor + critic) plus the optimiser state.
- LR and clip-`ε` annealing are linear in `_update_step / _total_updates`, where
  `_total_updates` is derived from **`total_frames`** in `setup()`. The Breakout
  experiment runs `trainer.total_frames = 100_000`, but `configs/algorithm/ppo.yaml`
  does **not** set `total_frames`, so the algorithm keeps its `40_000_000` default
  and the schedule progresses only ~0.25 % — `lr` and `ε` stay effectively
  constant. To make annealing meaningful on a short run, set `total_frames` on the
  algorithm to match the trainer's budget.

## Hyperparameters

Defaults on `PPOAlgorithm.__init__`:

| Param | Default | Meaning |
|-------|---------|---------|
| `lr` | `2.5e-4` | Adam learning rate |
| `eps` | `1e-6` | Adam epsilon |
| `weight_decay` | `0.0` | Adam weight decay |
| `max_grad_norm` | `0.5` | global gradient-norm clip |
| `anneal_lr` | `true` | linearly decay the learning rate to 0 |
| `gamma` | `0.99` | discount factor γ |
| `gae_lambda` | `0.95` | GAE bias–variance trade off λ |
| `clip_epsilon` | `0.1` | PPO clip range ε |
| `anneal_clip_epsilon` | `true` | linearly decay ε to 0 |
| `critic_coeff` | `1.0` | weight of the value loss |
| `entropy_coeff` | `0.01` | weight of the entropy bonus |
| `loss_critic_type` | `l2` | value loss type |
| `ppo_epochs` | `3` | optimisation epochs per collected batch |
| `mini_batch_size` | `1024` | minibatch size within each epoch |
| `frames_per_batch` | `4096` | rollout length per update |
| `init_random_frames` | `0` | warm up frames before learning starts |
| `total_frames` | `40_000_000` | training budget; also sizes the anneal schedule |
| `max_frames_per_traj` | `-1` | no per trajectory length cap |

**Breakout experiment overrides** (`configs/algorithm/ppo.yaml`): `clip_epsilon: 0.2`,
`critic_coeff: 0.25`, `entropy_coeff: 0.003`, `ppo_epochs: 6`,
`mini_batch_size: 512`, `frames_per_batch: 2048`. (`init_random_frames` and
`total_frames` are left at their `__init__` defaults — see the annealing note above.)

## Environment

Pixel Breakout (`ALE/Breakout-v5`, gymnasium backend, `frame_skip = 4`,
categorical actions). The transform stack: random no-op reset (up to 30),
grayscale, 84×84 resize, 4-frame stacking (`CatFrames`, so motion and ball
direction become observable), a per-episode step limit (4500), reward sum, and
float casting.

Training and evaluation use **separate** env configs via the Hydra package
override. The **training** env additionally applies `EndOfLifeTransform` each
lost life ends the episode, which densifies the learning signal, while the
**evaluation** env omits it so episode returns reflect full games. This is the
template's standard Atari convention:

```yaml
# configs/experiment/ppo/breakout.yaml
defaults:
  - override /algorithm: ppo
  - override /environment: breakout_train
  - override /environment@eval_environment: breakout_eval
```

## Quick start

```shell
python src/train.py experiment=ppo/breakout
```

The reference experiment runs 100k frames on CPU with 2 parallel envs and logs
to W&B.

## Results

Tracked on [W&B (LatentLab / IntroDRL26)](https://wandb.ai/LatentLab/IntroDRL26). <br>
https://wandb.ai/polinato-opencampus/torchrl-hydra-template

| Environment | Frames | Mean episode reward | Hardware |
|-------------|--------|---------------------|----------|
| ALE/Breakout-v5 | 100k (validation run) | ≈ 2.0–2.5 (near random) | CPU, 2 envs |

> **Status.** The 100k-frame CPU run validates the pipeline end to end, the
> critic loss decreases and policy entropy falls, so learning is happening, but
> it is far too short to actually solve Breakout, which needs on the order of
> millions of frames (and benefits from GPU + more parallel envs). A full
> benchmark is pending.

## References

- Schulman et al. (2017), *Proximal Policy Optimization Algorithms.* [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- Schulman et al. (2016), *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)
- Mnih et al. (2015), *Human-level control through deep reinforcement learning.* (Nature-CNN + Atari preprocessing)
