from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from framework.replay_buffer import RolloutBuffer


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def gen_net(envs, hidden_dim, layers, output_dim, std):
    net = [
        layer_init(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), hidden_dim),
        ),
        nn.Tanh(),
    ]
    for _ in range(layers):
        net.append(layer_init(nn.Linear(hidden_dim, hidden_dim)))
        net.append(nn.Tanh())
    net.append(layer_init(nn.Linear(hidden_dim, output_dim), std))

    return net


class Agent(nn.Module):
    def __init__(self, envs, hidden_dim, layers):
        super().__init__()
        self.critic = nn.Sequential(*gen_net(envs, hidden_dim, layers, 1, std=0.0))
        self.actor_mean = nn.Sequential(
            *gen_net(
                envs,
                hidden_dim,
                layers,
                np.prod(envs.single_action_space.shape),
                std=0.01,
            )
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.single_action_space.shape))
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )

    def get_action(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, 0, 0  # in order to match the scheme of actor.get_action

    def select_action(self, obs, device):
        with torch.no_grad():
            action, logprob, _, value = self.get_action_and_value(
                torch.Tensor(obs).to(device)
            )
            value = value.detach().cpu().numpy().flatten()
        action = action.detach().cpu().numpy()
        logprob = logprob.detach().cpu().numpy()
        return action, logprob, value

    def train_agent(self, rb: RolloutBuffer, args, optimizer) -> Dict[str, float]:
        buffer = rb.get()
        b_obs = buffer.observations
        b_logprobs = buffer.log_probs
        b_actions = buffer.actions
        b_advantages = buffer.advantages
        b_returns = buffer.returns
        b_values = buffer.values

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        return {
            "value_loss": v_loss.item(),
            "policy_loss": pg_loss.item(),
            "entropy": entropy_loss.item(),
            "old_approx_kl": old_approx_kl.item(),
            "approx_kl": approx_kl.item(),
            "clipfracs": clipfracs,
            "explained_variance": explained_var,
        }
