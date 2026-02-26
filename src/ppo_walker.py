import torch
from dm_control import suite,viewer
import numpy as np
import scipy.signal
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam


class uth_t(nn.Module):
    def __init__(s,xdim,udim,
                 hdim=32,fixed_var=True):
        super().__init__()
        s.xdim,s.udim = xdim, udim
        s.fixed_var=fixed_var

        ### TODO

        # policy mean network (2 layer MLP)
        s.mu_net = nn.Sequential(
            nn.Linear(xdim,hdim), # input feature --> hidden
            nn.Tanh(), # non-linearity
            nn.Linear(hdim,hdim), # deeper layer
            nn.Tanh(), # non-linearity
            nn.Linear(hdim,udim), # hidden --> output layer giving mu
        )

        # policy variance
        if fixed_var: # learnable fixed log standard vector
            s.log_std = nn.Parameter(th.zeros(udim))
        else: # state-dependent log standard vector
            s.std_net = nn.Sequential(
                nn.Linear(xdim, hdim),
                nn.Tanh(),
                nn.Linear(hdim, hdim),
                nn.Tanh(),
                nn.Linear(hdim, udim),  # output layer giving log-standard for each action dimension
            )

        # critic (value network)
        s.value_network = nn.Sequential(
            nn.Linear(xdim, hdim),
            nn.Tanh(),
            nn.Linear(hdim, hdim),
            nn.Tanh(),
            nn.Linear(hdim, 1)
        )

        ### END TODO

    def forward(s,x):
        ### TODO
        mu = s.mu_net(x)
        if s.fixed_var:
            std = s.log_std.exp().expand_as(mu) # std = exp(log_std)
        else:
            std = s.std_net(x).exp()            # std = exp(std_net)
        ### END TODO
        return mu,std

    def value(s, x):
        return s.value_network(x).squeeze()

def rollout(env,policy, T=1000):
    """
    e: environment
    policy: policy function
    value_function: value function
    T: time-steps
    """

    traj=[] # collects transitions
    t=env.reset()
    x=t.observation
    obs=np.array(x['orientations'].tolist()+[x['height']]+x['velocity'].tolist()) # observation state vector: orientations 14, height 1, velocities 9

    for _ in range(T):
        with th.no_grad():
            # gaussian parameters
            state = torch.as_tensor(obs, dtype = torch.float32).unsqueeze(0)
            mu,std = policy(state)
            distribution = Normal(mu, std)
            action = distribution.sample()[0].numpy()
            log_prob = distribution.log_prob(distribution.sample()).sum(axis=-1).item()
            value_func = policy.value(th.from_numpy(obs).float().unsqueeze(0)).item()

        next_t = env.step(action)
        next_x = next_t.observation
        reward, done = next_t.reward, next_t.last()
        next_obs = np.array(next_x['orientations'].tolist()+[next_x['height']]+next_x['velocity'].tolist())

        t=dict(obs=obs, next_obs=next_obs, action=action, reward=next_t.reward, mu=mu, done=next_t.last(), log_prob=log_prob, value=value_func)
        traj.append(t)
        obs = next_obs
        if next_t.last():
            break
    return traj


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class PPOBuffer:
    def __init__(s, obs_dim, act_dim, size=1e6, discount=0.99, lam=0.95):
        s.obs_buffer = np.zeros((size, obs_dim), dtype=np.float32) # states
        s.reward_buffer = np.zeros(size, dtype=np.float32) # rewards
        s.action_buffer = np.zeros((size, act_dim), dtype=np.float32) # actions
        s.log_prob_buffer = np.zeros(size, dtype=np.float32) # need for importance sampling ratio
        s.value_buffer = np.zeros(size, dtype=np.float32) # stores critic value estimate
        s.advantage_buffer = np.zeros(size, dtype=np.float32) # advantage estimates using rewards & values
        s.reward_to_go_buffer = np.zeros(size, dtype=np.float32) # targets --> regress value function toward these
        s.discount, s.lam = discount, lam # discount factor, gae param
        s.pointer, s.path_start, s.max_size = 0, 0, size


    def store(s, obs, act, reward, value, logp):
        index = s.pointer
        s.obs_buffer[index] = obs
        s.reward_buffer[index] = reward
        s.action_buffer[index] = act
        s.log_prob_buffer[index] = logp
        s.value_buffer[index] = value
        s.pointer += 1

    def finish_path(s, last_value = 0): # after traj ends
        sliced_path = slice(s.path_start, s.pointer)
        rewards = np.append(s.reward_buffer[sliced_path], last_value)
        values = np.append(s.value_buffer[sliced_path], last_value)

        ## gae
        deltas = rewards[:-1] + s.discount * values[1:] - values[:-1]
        s.advantage_buffer[sliced_path] = discount_cumsum(deltas, s.discount * s.lam)

        ## rewards to go for target vals
        s.reward_to_go_buffer[sliced_path] = discount_cumsum(rewards, s.discount)[:-1]
        s.path_start = s.pointer

    def get(s):
        assert s.pointer == s.max_size
        s.pointer, s.path_start = 0, 0

        advantage_mean, advantage_std = np.mean(s.advantage_buffer), np.std(s.advantage_buffer)+1e-8
        s.advantage_buffer = (s.advantage_buffer-advantage_mean)/advantage_std
        data = dict(
            obs = s.obs_buffer,
            action = s.action_buffer,
            reward_to_go = s.reward_to_go_buffer,
            advantage = s.advantage_buffer,
            log_prob = s.log_prob_buffer,
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

def train_ppo(env = 'walker',
              task = 'walk',
              x_dim = 14+1+9,
              u_dim = 6,                        # action vector dim
              hdim = 32,                        # hidden layer dim
              steps_per_epoch = 4000,
              epochs = 100,
              discount = 0.99,
              lam = 0.95,
              clip_ratio = 0.2,
              pi_lr = 3e-4,                     # policy LR
              vf_lr = 1e-3,                     # value func LR
              train_pi_iters = 80,
              train_vf_iters = 80,
              max_episode_length = 1000,
              target_kl = 0.01,
              fixed_policy=True,                # fixed policy std or state dependent
              seed = 0):
        np.random.seed(seed)
        torch.manual_seed(seed)

        timestep_list = []
        avg_returns = []
        std_returns = []

        env = suite.load(env, task, task_kwargs={'random':np.random.RandomState(seed)})

        # actor-critic
        actor_critic = uth_t(x_dim, u_dim, hdim, fixed_var=fixed_policy)
        pi_optimizer = Adam(actor_critic.mu_net.parameters(), lr=pi_lr)

        if fixed_policy:
            pi_optimizer.add_param_group({'params': actor_critic.log_std}) # add log standard to optimizer if fixed
        else:
            pi_optimizer.add_param_group({'params': actor_critic.std_net.parameters()}) # add std_net to optimizer if not fixed
        vf_optimizer = Adam(actor_critic.value_network.parameters(), lr=vf_lr)

        buffer = PPOBuffer(x_dim, u_dim, steps_per_epoch, discount, lam)

        # train
        for epoch in range(epochs):
            t = env.reset()
            obs = np.array(t.observation['orientations'].tolist() +
                           [t.observation['height']] +
                           t.observation['velocity'].tolist())
            episode_returns = []
            episode_return = 0
            episode_length = 0

            for s in range(steps_per_epoch): ## collect one epoch
                state_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                mu, std = actor_critic(state_tensor)
                distribution = Normal(mu, std)
                action_tensor = distribution.sample()  # (1, u_dim)
                action = action_tensor[0].numpy()
                log_prob = distribution.log_prob(action_tensor).sum(dim=-1).item()
                value = actor_critic.value(state_tensor).item()

                next_t = env.step(action)
                reward, done = next_t.reward, next_t.last()
                next_obs = np.array(next_t.observation['orientations'].tolist() +
                                    [next_t.observation['height']] +
                                    next_t.observation['velocity'].tolist())

                buffer.store(obs, action, reward, value, log_prob) # store in buffer

                episode_return += reward
                episode_length += 1
                obs = next_obs

                if done or (episode_length == max_episode_length):
                    episode_returns.append(episode_return)
                    last_val = 0 if done else actor_critic.value(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).item()
                    buffer.finish_path(last_val)

                    t = env.reset()
                    obs = np.array(t.observation['orientations'].tolist() +
                           [t.observation['height']] +
                           t.observation['velocity'].tolist())
                    episode_return = 0
                    episode_length = 0

            timestep_list.append((epoch+1) * steps_per_epoch)
            avg_returns.append(np.mean(episode_returns))
            std_returns.append(np.std(episode_returns))
            data = buffer.get()

            for i in range(train_pi_iters):
                mu, std = actor_critic(data['obs'])
                distribution = Normal(mu, std)
                log_prob = distribution.log_prob(data['action']).sum(axis=-1)
                ratio = torch.exp(log_prob - data['log_prob'])
                clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * data['advantage']
                loss_pi = -torch.min(ratio * data['advantage'], clip_adv).mean() ## PPO LOSS
                kl = (data['log_prob'] - log_prob).mean().item()
                if kl > 1.5 * target_kl:
                    break
                pi_optimizer.zero_grad()
                loss_pi.backward()
                pi_optimizer.step()


            for _ in range(train_vf_iters):
                v_est = actor_critic.value(data['obs'])
                loss_v = F.mse_loss(v_est, data['reward_to_go'])
                vf_optimizer.zero_grad()
                loss_v.backward()
                vf_optimizer.step()

            print(f"Epoch {epoch+1}/ {epochs}! done :)")

        np.savez('ppo_returns.npz',
                 timesteps=np.array(timestep_list),
                 avg_returns=np.array(avg_returns),
                 std_returns=np.array(std_returns))
        return actor_critic

if __name__ == '__main__':
    actor_critic = train_ppo(steps_per_epoch=5000, epochs=250)
    viz_env = suite.load('walker', 'walk', {'random': np.random.RandomState(0)})


    def playback_policy(time_step):
        obs = time_step.observation
        x = np.array(obs['orientations'].tolist() +
                     [obs['height']] +
                     obs['velocity'].tolist())
        with torch.no_grad():
            state = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
            mu, _ = actor_critic(state)
            action = mu[0].numpy()
        return action


    viewer.launch(viz_env, playback_policy)
















































"""
Setup walker environment
r0 = np.random.RandomState(42)
e = suite.load('walker', 'walk',
                 task_kwargs={'random': r0})
U=e.action_spec();udim=U.shape[0];
X=e.observation_spec();xdim=14+1+9;

#Visualize a random controller
def u(dt):
    return np.random.uniform(low=U.minimum,
                             high=U.maximum,
                             size=U.shape)
viewer.launch(e,policy=u)


# Example rollout using a network
uth=uth_t(xdim,udim)
traj=rollout(e,uth)
"""