import gymnasium as gym
from tqdm import tqdm

env = gym.make("LunarLander-v3", render_mode="rgb_array")
observation, info = env.reset(seed=42)

print("initial observation: ", observation)

action_samples = []
observation_samples = []
n_episodes = 100

for episode in tqdm(range(n_episodes)):
    #one episode
    episode_over = False
    while not episode_over:
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated
        # action_samples.append(env.action_space.sample())
        # observation_samples.append(env.observation_space.sample)

env.close()