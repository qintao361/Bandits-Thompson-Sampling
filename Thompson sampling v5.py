import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

NUM_TRIALS = 2000
BAND_PROB = [0.2, 0.5, 0.75]

class Bandit:
    def __init__(self, p):
        self.p = p
        self.a = 1
        self.b = 1
        self.N = 0

    def pull(self):
        return np.random.rand() < self.p

    def sample(self):
        return np.random.beta(self.a, self.b)

    def update(self, x):
        self.N += 1
        self.a += x
        self.b += 1-x

def plot(bandits, trial):
    x = np.linspace(0,1,200)
    for b in bandits:
        y = beta.pdf(x, b.a, b.b)
        label = f"real p: {b.p}, win rate: {b.a - 1}/{b.N}"
        plt.plot(x, y, label=label)

    plt.title(f"Bandit distributions after {trial} trials")
    plt.legend()
    plt.show()


def experiment():
    bandits = [Bandit(p) for p in BAND_PROB]
    sample_points = [5, 10, 15, 20, 50, 100, 200, 500, 1000, 1500, 1999]
    rewards = []

    for i in range(NUM_TRIALS):
        j = np.argmax([b.sample() for b in bandits])

        #plot the posteriors
        if i in sample_points:
            plot(bandits, i)

        x = bandits[j].pull()
        rewards.append(x)
        bandits[j].update(x)

    print("Total reward:", np.sum(rewards))
    print("Overall win rate:", np.sum(rewards)/NUM_TRIALS)
    print("num of times selected each bandit", [b.N for b in bandits])

if __name__ == '__main__':
    experiment()

