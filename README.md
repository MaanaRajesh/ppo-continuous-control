# ppo-continuous-control
Proximal Policy Optimization (PPO) in PyTorch for continuous control in MuJoCo (DM Control Walker). Includes GAE, clipped objective, value function learning, and performance visualization.

# PPO for Continuous Control in MuJoCo (Walker)

Implementation of **Proximal Policy Optimization (PPO)** in PyTorch for continuous control using DeepMind Control Suite (MuJoCo).  
The agent is trained on the **Walker: walk** task using clipped surrogate loss and Generalized Advantage Estimation (GAE).

---

## Overview

This project implements an actor–critic PPO framework from scratch, including:

- Gaussian policy with learned mean and variance
- Clipped PPO objective
- Generalized Advantage Estimation (GAE-λ)
- Value function regression with reward-to-go targets
- KL-based early stopping
- Performance logging and visualization

The agent is trained for over **1M+ environment interactions**, and learning stability, return trends, and variance are analyzed.

---

## Environment

- **DeepMind Control Suite**
- Task: `walker / walk`
- Continuous 6-DoF action space
- Observation: orientations (14) + height (1) + velocities (9)

---

## Architecture

### Policy Network (Actor)

- 2-layer MLP (Tanh activations)
- Outputs:
  - Mean vector μ(s)
  - Log standard deviation (fixed or state-dependent)
- Action distribution:  
  `π(a|s) = Normal(μ(s), σ(s))`

### Value Network (Critic)

- 2-layer MLP
- Outputs scalar state value V(s)

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Steps per epoch | 5000 |
| Epochs | 250 |
| Discount (γ) | 0.99 |
| GAE λ | 0.95 |
| Clip ratio | 0.2 |
| Policy LR | 3e-4 |
| Value LR | 1e-3 |
| Target KL | 0.01 |

---

## PPO Objective

The clipped objective:

\[
L^{CLIP} = \mathbb{E} \left[ \min \left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
\]

where

\[
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
\]

Advantages are computed using **GAE-λ**.

---

## Results

Training performance is logged in `ppo_returns.npz` and visualized with `plot_ppo_returns.py`.

The Walker learns stable locomotion behavior over training, with increasing average returns and reduced variance across episodes.

