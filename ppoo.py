import gymnasium as gym

from stable_baselines3 import dqn

# env = gym.make("CartPole-v1", render_mode="rgb_array")
# env = gym.make("ALE/Breakout-v5", render_mode="rgb_array") # test 1
env = gym.make("ALE/Pong-v5", render_mode="rgb_array") # test 2
# env = gym.make("Reacher-v4", render_mode="rgb_array") # test 3


model = dqn("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=1e7)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    # vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()