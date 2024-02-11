import matplotlib.pyplot as plt

# Data
environments = ['worlds_unique', 'worlds_train', 'worlds_test\n(TRAIN)']
success_rates = [37.04414367675781, 51.0805549621582, 91.0]
average_rewards = [11.634363014272445, 15.339895911675773, 25.104485350131988]
average_steps = [27.66602687140115, 31.93909626719057, 37.666]

# Creating bar plots
x = range(len(environments))

plt.figure(figsize=(12, 6))

# Subplot for Success Rate
plt.subplot(1, 3, 1)
plt.bar(x, success_rates, color='blue')
plt.xticks(x, environments)
plt.ylabel('Success Rate (%)')
plt.title('Success Rate by Environment')

# Subplot for Average Reward
plt.subplot(1, 3, 2)
plt.bar(x, average_rewards, color='green')
plt.xticks(x, environments)
plt.ylabel('Average Reward')
plt.title('Average Reward by Environment')

# Subplot for Average Steps
plt.subplot(1, 3, 3)
plt.bar(x, average_steps, color='red')
plt.xticks(x, environments)
plt.ylabel('Average Steps')
plt.title('Average Steps by Environment')

# Show the plot
plt.tight_layout()
plt.savefig('results.png')
plt.show()

