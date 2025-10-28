from __future__ import absolute_import, division, print_function

import os
import random
from collections import deque, namedtuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 

from smac.env import StarCraft2Env

LR = 1e-4
EPSILON = 0.3
GAMMA = 0.99
BATCH_SIZE = 64
REPLAY_CAPACITY = 50000
TRAIN_START = 1000
TARGET_UPDATE_INTERVAL = 200
MAX_TRAIN_STEPS = 200000
MAX_EPISODES = 5000
MAX_EPISODE_LEN = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_INTERVAL = 5

Transition = namedtuple("Transition", ("obs", "state", "action", "reward", "next_obs", "next_state", "done", "avail_mask", "next_avail_mask"))

class AgentNetwork(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_dim=64):
        super(AgentNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


class MixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim, hidden_dim=64, hyper_hidden=128):
        super(MixingNetwork, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, n_agents * hidden_dim)
        )
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, hidden_dim)
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, agent_qs, state):
        batch_size = agent_qs.size(0)

        w1 = self.hyper_w1(state)  # (B, n_agents * hidden_dim)
        w1 = w1.view(batch_size, self.n_agents, self.hidden_dim)  # (B, n_agents, H)
        b1 = self.hyper_b1(state).view(batch_size, 1, self.hidden_dim)  # (B,1,H)

        # (B,1,n_agents) x (B,n_agents,H) -> (B,1,H)
        agent_qs = agent_qs.view(batch_size, 1, self.n_agents)
        hidden = torch.relu(torch.bmm(agent_qs, w1) + b1)  # (B,1,H)

        w2 = self.hyper_w2(state).view(batch_size, self.hidden_dim, 1)  # (B,H,1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)  # (B,1,1)

        y = torch.bmm(hidden, w2) + b2  # (B,1,1)
        q_total = y.view(batch_size)  # (B,)
        return q_total

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs = np.array([t.obs for t in batch], dtype=np.float32)            # (B, n_agents, obs_dim)
        state = np.array([t.state for t in batch], dtype=np.float32)        # (B, state_dim)
        actions = np.array([t.action for t in batch], dtype=np.int64)       # (B, n_agents)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)     # (B,)
        next_obs = np.array([t.next_obs for t in batch], dtype=np.float32)
        next_state = np.array([t.next_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)         # (B,)
        avail = np.array([t.avail_mask for t in batch], dtype=np.float32)   # (B, n_agents, n_actions)
        next_avail = np.array([t.next_avail_mask for t in batch], dtype=np.float32)

        return obs, state, actions, rewards, next_obs, next_state, dones, avail, next_avail

    def __len__(self):
        return len(self.buffer)

def choose_actions(agent_net, obs_batch, avail_actions_batch, epsilon, device):
    n_agents = len(obs_batch)
    actions = []
    for i in range(n_agents):
        avail_mask = np.array(avail_actions_batch[i], dtype=np.int32)
        avail_inds = np.nonzero(avail_mask)[0]
        if len(avail_inds) == 0:
            actions.append(0)
            continue
        if random.random() < epsilon:
            actions.append(int(np.random.choice(avail_inds)))
        else:
            obs_t = torch.tensor(obs_batch[i], dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q = agent_net(obs_t).squeeze(0).cpu().numpy()
            # mask
            q_masked = q.copy()
            q_masked[avail_mask == 0] = -1e9
            # greedy argmax
            a = int(np.argmax(q_masked))
            actions.append(a)
    return actions

def train_step(agent_net, target_agent_net, mixer, target_mixer, optimizer, buffer):
    if len(buffer) < BATCH_SIZE:
        return None

    obs_b, state_b, actions_b, rewards_b, next_obs_b, next_state_b, dones_b, avail_b, next_avail_b = buffer.sample(BATCH_SIZE)

    obs = torch.tensor(obs_b, dtype=torch.float32, device=DEVICE)
    state = torch.tensor(state_b, dtype=torch.float32, device=DEVICE)
    actions = torch.tensor(actions_b, dtype=torch.long, device=DEVICE)
    rewards = torch.tensor(rewards_b, dtype=torch.float32, device=DEVICE)
    next_obs = torch.tensor(next_obs_b, dtype=torch.float32, device=DEVICE)
    next_state = torch.tensor(next_state_b, dtype=torch.float32, device=DEVICE)
    dones = torch.tensor(dones_b, dtype=torch.float32, device=DEVICE)
    avail = torch.tensor(avail_b, dtype=torch.float32, device=DEVICE)
    next_avail = torch.tensor(next_avail_b, dtype=torch.float32, device=DEVICE)

    B, n_agents, obs_dim = obs.size()
    n_actions = agent_net.out.out_features if hasattr(agent_net, 'out') else agent_net.out.weight.size(0)

    obs_flat = obs.view(B * n_agents, obs_dim)
    q_values = agent_net(obs_flat)  # (B*n_agents, n_actions)
    q_values = q_values.view(B, n_agents, -1)  # (B, n_agents, n_actions)

    # Gather Q of taken actions
    actions_onehot = actions.unsqueeze(-1)  # (B, n_agents, 1)
    q_taken = q_values.gather(dim=2, index=actions_onehot).squeeze(2)  # (B, n_agents)

    # compute target Q-values using target networks
    next_obs_flat = next_obs.view(B * n_agents, obs_dim)
    with torch.no_grad():
        next_q_values = target_agent_net(next_obs_flat).view(B, n_agents, -1)  # (B, n_agents, n_actions)
        # mask unavailable actions
        very_neg = -1e9
        next_q_values[next_avail == 0] = very_neg
        next_max_q = next_q_values.max(dim=2)[0]  # (B, n_agents)

    # mix per-agent Qs to total Q
    q_total = mixer(q_taken, state)  # (B,)
    with torch.no_grad():
        target_q_total = target_mixer(next_max_q, next_state)  # (B,)
        target = rewards + (1 - dones) * GAMMA * target_q_total

    loss_fn = nn.MSELoss()
    loss = loss_fn(q_total, target)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(agent_net.parameters()) + list(mixer.parameters()), 10.0)
    optimizer.step()

    return loss.item()

def main():
    env = StarCraft2Env(map_name="8m")
    env_info = env.get_env_info()
    n_agents = env_info['n_agents']
    n_actions = env_info['n_actions']
    obs_dim = env_info['obs_shape']
    state_dim = env_info['state_shape']

    print(f"SMAC map={env.map_name}, n_agents={n_agents}, n_actions={n_actions}, obs_dim={obs_dim}, state_dim={state_dim}")
    assert n_agents == 8, "This script expects 8 agents"

    agent_net = AgentNetwork(obs_dim, n_actions).to(DEVICE)
    target_agent_net = AgentNetwork(obs_dim, n_actions).to(DEVICE)
    target_agent_net.load_state_dict(agent_net.state_dict())

    mixer = MixingNetwork(n_agents, state_dim).to(DEVICE)
    target_mixer = MixingNetwork(n_agents, state_dim).to(DEVICE)
    target_mixer.load_state_dict(mixer.state_dict())

    optimizer = optim.Adam(list(agent_net.parameters()) + list(mixer.parameters()), lr=LR)

    replay = ReplayBuffer(REPLAY_CAPACITY)

    global_step = 0
    grad_steps = 0

    for ep in range(MAX_EPISODES):
        env.reset()
        terminated = False
        episode_reward = 0.0
        step_in_ep = 0

        obs = env.get_obs()  # list of per-agent obs
        obs = [np.array(o, dtype=np.float32) for o in obs]
        state = np.array(env.get_state(), dtype=np.float32)

        while not terminated and step_in_ep < MAX_EPISODE_LEN:
            avail_masks = [np.array(env.get_avail_agent_actions(a), dtype=np.int32) for a in range(n_agents)]
            actions = choose_actions(agent_net, obs, avail_masks, EPSILON, DEVICE)

            reward, terminated, info = env.step(actions)
            next_obs = env.get_obs()
            next_obs = [np.array(o, dtype=np.float32) for o in next_obs]
            next_state = np.array(env.get_state(), dtype=np.float32)

            replay.push(
                np.stack(obs, axis=0),  # obs: (n_agents, obs_dim)
                state,                  # global state
                np.array(actions, dtype=np.int64),
                float(reward),
                np.stack(next_obs, axis=0),
                next_state,
                float(terminated),
                np.stack(avail_masks, axis=0),
                np.stack([np.array(env.get_avail_agent_actions(a), dtype=np.int32) for a in range(n_agents)], axis=0)
            )

            obs = next_obs
            state = next_state
            episode_reward += reward
            step_in_ep += 1
            global_step += 1

            if global_step > TRAIN_START:
                loss = train_step(agent_net, target_agent_net, mixer, target_mixer, optimizer, replay)
                grad_steps += 1

                if grad_steps % TARGET_UPDATE_INTERVAL == 0:
                    target_agent_net.load_state_dict(agent_net.state_dict())
                    target_mixer.load_state_dict(mixer.state_dict())

                if grad_steps >= MAX_TRAIN_STEPS:
                    break

        if ep % PRINT_INTERVAL == 0:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Ep {ep} | Reward={episode_reward:.2f} | Replay={len(replay)} | GradSteps={grad_steps}")

        if grad_steps >= MAX_TRAIN_STEPS:
            print("Reached max training steps. Exit.")
            break

    # save
    os.makedirs('models', exist_ok=True)
    torch.save({'agent_net': agent_net.state_dict(), 'mixer': mixer.state_dict()}, 'models/qmix_shared.pth')
    print("Saved models/models/qmix_shared.pth")

    env.close()


if __name__ == '__main__':
    main()
