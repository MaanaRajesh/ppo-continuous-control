import numpy as np
import matplotlib.pyplot as plt

data = np.load('ppo_returns.npz')

timesteps = data['timesteps']
average_returns = data['avg_returns']

plt.figure()
plt.plot(timesteps, average_returns, marker='o')
plt.xlabel('Timesteps')
plt.ylabel('Average Return')
plt.title('PPO on DM Control Walker')
plt.grid(True)
plt.tight_layout()
plt.show()