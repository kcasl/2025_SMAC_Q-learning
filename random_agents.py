from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env import StarCraft2Env
import numpy as np
import os
from datetime import datetime

def main():
    env = StarCraft2Env(map_name="8m")

    env.replay_dir = r"D:\StarCraft II\Replays"
    os.makedirs(env.replay_dir, exist_ok=True)

    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]

    n_episodes = 1000

    for e in range(n_episodes):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        env.replay_prefix = f"{env.map_name}_ep{e}_{timestamp}"

        env.reset()
        terminated = False
        episode_reward = 0

        while not terminated:
            obs = env.get_obs()
            state = env.get_state()
            # env.render()  # Uncomment for rendering

            actions = []
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.choice(avail_actions_ind)
                actions.append(action)

            reward, terminated, _ = env.step(actions)
            episode_reward += reward

        print("Total reward in episode {} = {}".format(e, episode_reward))

    try:
        env.save_replay()
        print(f"Replay saved (prefix={env.replay_prefix}) in {env.replay_dir}")
    except Exception as exc:
        print("Replay save failed:", exc)

    env.close()


if __name__ == "__main__":
    main()
