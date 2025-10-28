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

SEED = 42
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 128
BUFFER_CAPACITY = 2000000
MIN_REPLAY_SIZE = 20000
TARGET_UPDATE_FREQ = 1000      # env steps
TRAIN_FREQ = 4                 # train every N env steps
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 1e-5               # per env step
MAX_EPISODES = 200000
MAX_STEPS_EPISODE = 200        # safety cap
DEVICE = torch.device("cpu")


Transition = namedtuple("Transition", ("obs", "action", "reward", "next_obs", "done", "avail", "next_avail"))

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=128):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)
    
def select_action_epsilon_greedy(net, obs_np, avail_actions, epsilon):
    # obs_np: numpy array shape (obs_dim,)
    obs = torch.from_numpy(obs_np).float().unsqueeze(0).to(DEVICE)  # (1, obs_dim)
    with torch.no_grad():
        qvals = net(obs).cpu().numpy().squeeze(0)  # (n_actions,)
    avail_idx = np.nonzero(avail_actions)[0]
    if np.random.rand() < epsilon:
        return int(np.random.choice(avail_idx))
    # mask unavailable
    masked = np.full_like(qvals, -1e9)
    masked[avail_idx] = qvals[avail_idx]
    return int(np.argmax(masked))


def compute_td_loss(net, target_net, transitions, n_actions, gamma):
    obs_batch = np.asarray(transitions.obs)            # (batch, obs_dim)
    action_batch = np.asarray(transitions.action)      # (batch,)
    reward_batch = np.asarray(transitions.reward)      # (batch,)
    next_obs_batch = np.asarray(transitions.next_obs)  # (batch, obs_dim)
    done_batch = np.asarray(transitions.done).astype(np.float32)
    avail_batch = np.asarray(transitions.avail)        # (batch, n_actions)
    next_avail_batch = np.asarray(transitions.next_avail)

    obs_t = torch.from_numpy(obs_batch).float().to(DEVICE)
    actions_t = torch.from_numpy(action_batch).long().to(DEVICE).unsqueeze(1)
    rewards_t = torch.from_numpy(reward_batch).float().to(DEVICE).unsqueeze(1)
    next_obs_t = torch.from_numpy(next_obs_batch).float().to(DEVICE)
    done_t = torch.from_numpy(done_batch).float().to(DEVICE).unsqueeze(1)
    avail_t = torch.from_numpy(avail_batch).float().to(DEVICE)
    next_avail_t = torch.from_numpy(next_avail_batch).float().to(DEVICE)

    q_values = net(obs_t)  # (batch, n_actions)
    q_a = q_values.gather(1, actions_t)  # (batch, 1)

    with torch.no_grad():
        q_next = target_net(next_obs_t)  # (batch, n_actions)
        # mask next unavailable actions
        neg_inf = torch.full_like(q_next, -1e9)
        q_next_masked = torch.where(next_avail_t > 0.5, q_next, neg_inf)
        q_next_max = q_next_masked.max(dim=1, keepdim=True)[0]  # (batch,1)
        target = rewards_t + gamma * (1.0 - done_t) * q_next_max

    loss = nn.functional.mse_loss(q_a, target)
    return loss


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = StarCraft2Env(map_name="8m")
    env.replay_dir = r"D:\StarCraft II\Replays"
    os.makedirs(env.replay_dir, exist_ok=True)

    env_info = env.get_env_info()
    n_agents = env_info["n_agents"]
    n_actions = env_info["n_actions"]
    obs_dim = env.get_obs_size()

    policy_net = QNetwork(obs_dim, n_actions).to(DEVICE)
    target_net = QNetwork(obs_dim, n_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay = ReplayBuffer(BUFFER_CAPACITY)

    total_env_steps = 0
    epsilon = EPS_START

    print("Warmup replay buffer...")
    while len(replay) < MIN_REPLAY_SIZE:
        env.reset()
        terminated = False
        step = 0
        while not terminated and step < MAX_STEPS_EPISODE:
            obs_list = env.get_obs()
            actions = []
            for aid in range(n_agents):
                avail = env.get_avail_agent_actions(aid)
                avail_idx = np.nonzero(avail)[0]
                actions.append(int(np.random.choice(avail_idx)))
            reward, terminated, _ = env.step(actions)
            next_obs_list = env.get_obs()
            for aid in range(n_agents):
                replay.push(obs_list[aid], actions[aid], reward, next_obs_list[aid], terminated, env.get_avail_agent_actions(aid), env.get_avail_agent_actions(aid))
            step += 1

    print("Warmup done. Starting training...")

    episode_rewards = []
    for ep in range(MAX_EPISODES):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        env.replay_prefix = f"{env.map_name}_ep{ep}_{timestamp}"

        env.reset()
        terminated = False
        ep_reward = 0
        step_in_ep = 0

        while not terminated and step_in_ep < MAX_STEPS_EPISODE:
            obs_list = env.get_obs()
            actions = []
            avails = []
            for aid in range(n_agents):
                avail = env.get_avail_agent_actions(aid)
                avails.append(avail)
                act = select_action_epsilon_greedy(policy_net, np.array(obs_list[aid]), np.array(avail), epsilon)
                actions.append(act)

            reward, terminated, _ = env.step(actions)
            next_obs_list = env.get_obs()
            next_avails = [env.get_avail_agent_actions(aid) for aid in range(n_agents)]

            for aid in range(n_agents):
                replay.push(obs_list[aid], actions[aid], reward, next_obs_list[aid], terminated, avails[aid], next_avails[aid])

            ep_reward += reward
            step_in_ep += 1
            total_env_steps += 1

            # train
            if total_env_steps % TRAIN_FREQ == 0 and len(replay) >= BATCH_SIZE:
                batch = replay.sample(BATCH_SIZE)
                loss = compute_td_loss(policy_net, target_net, batch, n_actions, GAMMA)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                optimizer.step()

            # target network update
            if total_env_steps % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # epsilon decay
            epsilon = max(EPS_END, epsilon - EPS_DECAY)

        episode_rewards.append(ep_reward)
        if ep % 10 == 0:
            avg_r = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 1 else ep_reward
            print(f"Ep {ep:05d} steps {total_env_steps} reward {ep_reward:.2f} avg100 {avg_r:.2f} eps {epsilon:.3f}")

        # save occasional checkpoint
        if ep % 500 == 0 and ep > 0:
            torch.save(policy_net.state_dict(), f"dqn_smac_policy_ep{ep}.pth")

    # final save
    torch.save(policy_net.state_dict(), "dqn_smac_policy_final.pth")
    try:
        env.save_replay()
    except Exception:
        pass
    env.close()


if __name__ == "__main__":
    main()
