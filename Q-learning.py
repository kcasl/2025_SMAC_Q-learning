from __future__ import absolute_import, division, print_function
from smac.env import StarCraft2Env
import numpy as np
import os
from datetime import datetime
from collections import defaultdict
import csv

def compstate(state, dims=3):
    state = np.array(state)
    chunks = np.array_split(state, dims)
    compressed = np.array([np.mean(c) for c in chunks] + [np.std(c) for c in chunks])
    return np.clip(compressed, -1, 1)

def discstate(state, bins=10):
    discrete = np.digitize(state, np.linspace(-1, 1, bins))
    return tuple(discrete)

def epsilongreedy(Q, state, epsilon, avail_actions):
    avail_actions = np.nonzero(avail_actions)[0]
    if np.random.rand() < epsilon:
        return np.random.choice(avail_actions)
    q_values = Q[state]
    masked_q = np.full_like(q_values, -np.inf)
    masked_q[avail_actions] = q_values[avail_actions]
    return np.argmax(masked_q)

def main():
    env = StarCraft2Env(map_name="8m")
    env.replay_dir = r"D:\StarCraft II\Replays"
    os.makedirs(env.replay_dir, exist_ok=True)
    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]

    alpha = 0.1
    gamma = 0.95
    epsilon = 1.0
    min_epsilon = 0.1
    epsilon_dec = 0.9998
    alpha_dec = 0.9995      

    n_episodes = 25000
    bins = 10
    dims = 3

    Q_tables = [defaultdict(lambda: np.zeros(n_actions)) for _ in range(n_agents)]
    rewards_log = []

    for ep in range(n_episodes):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        env.replay_prefix = f"{env.map_name}_ep{ep}_{timestamp}"
        obs = env.reset()
        state = env.get_state()
        terminated = False
        episode_reward = 0

        if ep % 1500 == 0:
            epsilon += 0.3

        while not terminated:
            compressed = compstate(state, dims=dims)
            discrete_state = discstate(compressed, bins=bins)

            actions = []
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                action = epsilongreedy(Q_tables[agent_id], discrete_state, epsilon, avail_actions)
                actions.append(action)

            reward, terminated, _ = env.step(actions)
            next_state = env.get_state()
            compressed_next = compstate(next_state, dims=dims)
            discrete_next = discstate(compressed_next, bins=bins)

            scaled_reward = reward / 10

            for agent_id in range(n_agents):
                a = actions[agent_id]
                Q = Q_tables[agent_id]
                q_val = Q[discrete_state][a]
                best_next = np.max(Q[discrete_next])
                target = scaled_reward + gamma * best_next
                Q[discrete_state][a] = (1 - alpha) * q_val + alpha * target

            state = next_state
            episode_reward += reward

        rewards_log.append(episode_reward)
        epsilon = max(min_epsilon, epsilon * epsilon_dec)
        alpha *= alpha_dec

        if ep % 100 == 0:
            avg_r = np.mean(rewards_log[-100:])
            print(f"[Ep {ep:05d}] AvgReward={avg_r:.3f} eps={epsilon:.3f} alp={alpha:.4f}")

    with open("q-learning_rewards.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for i, r in enumerate(rewards_log):
            writer.writerow([i, r])

    try:
        env.save_replay()
        print(f"Rep saved ({env.replay_prefix}) in {env.replay_dir}")
    except Exception as ex:
        print("Rep save failed", ex)
    env.close()


if __name__ == "__main__":
    main()