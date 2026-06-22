"""Proximal Policy Optimization (PPO).

Schulman et al. (2017), "Proximal Policy Optimization Algorithms."
https://arxiv.org/abs/1707.06347
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from torchrl.envs import EnvBase
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer, OneHot
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from src.algorithms.base import BaseAlgorithm, CollectorConfig, TrainingState
from src.networks import make_atari_ppo_actor, make_atari_ppo_critic


class PPOAlgorithm(BaseAlgorithm):

    def __init__(
        self,
        device: torch.device | None = None,
        *,

        actor_network: Callable[..., nn.Module] = make_atari_ppo_actor,
        critic_network: Callable[..., nn.Module] = make_atari_ppo_critic,

        obs_key: str = "pixels",

        # optimizer
        lr: float = 2.5e-4,
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        max_grad_norm: float = 0.5,
        anneal_lr: bool = True,

        # PPO loss
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.1,
        anneal_clip_epsilon: bool = True,
        critic_coeff: float = 1.0,
        entropy_coeff: float = 0.01,
        loss_critic_type: str = "l2",

        # training
        ppo_epochs: int = 3,
        mini_batch_size: int = 1024,

        # collection
        frames_per_batch: int = 4096,
        init_random_frames: int = 0,
        total_frames: int = 40_000_000,
        max_frames_per_traj: int = -1,
    ):
        super().__init__(device)

        self.obs_key = obs_key

        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.anneal_lr = anneal_lr

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.clip_epsilon = clip_epsilon
        self.anneal_clip_epsilon = anneal_clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.critic_coeff = critic_coeff
        self.loss_critic_type = loss_critic_type

        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

        self.frames_per_batch = frames_per_batch
        self.init_random_frames = init_random_frames
        self.total_frames = total_frames
        self.max_frames_per_traj = max_frames_per_traj

        self._make_actor = actor_network
        self._make_critic = critic_network

        self._update_step = 0

    # ----------------------------------------------------
    # Setup
    # ----------------------------------------------------

    def setup(self, make_env: Callable[[], EnvBase]):

        proof_env = make_env()

        obs_shape = tuple(
            proof_env.observation_spec[self.obs_key].shape
        )

        action_dim = proof_env.action_spec.space.n

        actor_net = self._make_actor(
            obs_shape,
            action_dim,
        ).to(self.device)

        self.actor = ProbabilisticActor(
            module=TensorDictModule(
                actor_net,
                in_keys=[self.obs_key],
                out_keys=["logits"],
            ),
            in_keys=["logits"],
            out_keys=["action"],
            distribution_class=torch.distributions.Categorical,
            return_log_prob=True,
        ).to(self.device)

        self.critic = ValueOperator(
            module=self._make_critic(
                obs_shape,
                action_dim,
            ),
            in_keys=[self.obs_key],
        ).to(self.device)

        self.loss_module = ClipPPOLoss(
            actor_network=self.actor,
            critic_network=self.critic,
            clip_epsilon=self.clip_epsilon,
            entropy_coeff=self.entropy_coeff,
            critic_coeff=self.critic_coeff,
            loss_critic_type=self.loss_critic_type,
            normalize_advantage=True,
        )

        self.adv_module = GAE(
            gamma=self.gamma,
            lmbda=self.gae_lambda,
            value_network=self.critic,
            device=self.device,
        )

        self.data_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(self.frames_per_batch, device=self.device),
            sampler=SamplerWithoutReplacement(),
            batch_size=self.mini_batch_size,
        )

        self.optimizer = torch.optim.Adam(
            self.loss_module.parameters(),
            lr=self.lr,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )

        num_batches = self.total_frames // self.frames_per_batch
        num_minibatches = self.frames_per_batch // self.mini_batch_size

        self._total_updates = (
            num_batches
            * self.ppo_epochs
            * num_minibatches
        )

    # ----------------------------------------------------
    # Collector
    # ----------------------------------------------------

    def get_collector_config(self):

        return CollectorConfig(
            frames_per_batch=self.frames_per_batch,
            init_random_frames=self.init_random_frames,
            max_frames_per_traj=self.max_frames_per_traj,
        )

    # ----------------------------------------------------
    # Training
    # ----------------------------------------------------

    def step(self, batch: TensorDict) -> dict[str, float]:

        batch = batch.to(self.device)
        batch = batch.reshape(-1)

        with torch.no_grad():
            batch = self.adv_module(batch)

        self.data_buffer.empty()
        self.data_buffer.extend(batch)

        actor_losses = []
        critic_losses = []
        entropy_losses = []

        for _ in range(self.ppo_epochs):
            for mb in self.data_buffer:
                mb = mb.to(self.device)
                loss = self.loss_module(mb)
                total_loss = loss["loss_objective"] + loss["loss_critic"]
                if "loss_entropy" in loss.keys():
                    total_loss = total_loss + loss["loss_entropy"]

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.loss_module.parameters(),
                    self.max_grad_norm,
                )

                if self.anneal_lr:
                    progress = self._update_step / self._total_updates
                    lr = self.lr * (1 - progress)

                    for group in self.optimizer.param_groups:
                        group["lr"] = lr

                if self.anneal_clip_epsilon:
                    progress = self._update_step / self._total_updates
                    self.loss_module.clip_epsilon.copy_(
                        torch.tensor(
                            self.clip_epsilon * (1 - progress),
                            device=self.device
                        )
                    )

                self._update_step += 1

                self.optimizer.step()

                actor_losses.append(loss["loss_objective"].detach())
                critic_losses.append(loss["loss_critic"].detach())
                entropy_losses.append(
                    loss["loss_entropy"].detach()
                    if "loss_entropy" in loss.keys()
                    else torch.tensor(0.0, device=self.device)
                )

        return {
            "train/loss_actor": torch.stack(actor_losses).mean().item(),
            "train/loss_critic": torch.stack(critic_losses).mean().item(),
            "train/loss_entropy": torch.stack(entropy_losses).mean().item(),
        }

    # ----------------------------------------------------
    # Policy
    # ----------------------------------------------------

    def get_policy(self):

        return self.actor

    def get_explore_policy(self):

        return self.actor

    # ----------------------------------------------------
    # Checkpoint
    # ----------------------------------------------------

    def _get_training_state(self):

        return TrainingState(
            step=0,
            policy_state_dict=self.loss_module.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
        )

    def _load_training_state(
        self,
        state: TrainingState,
    ):

        self.loss_module.load_state_dict(state.policy_state_dict)

        self.optimizer = torch.optim.Adam(
            self.loss_module.parameters(),
            lr=self.lr,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )
        if state.optimizer_state_dict is not None:
            self.optimizer.load_state_dict(state.optimizer_state_dict)
