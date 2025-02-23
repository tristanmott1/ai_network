from scipy.stats import norm, beta
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


a, b = 14, 6
x = np.linspace(0, 1, 1000)
sigma = 0.2
n_samples = 100000

for i, s in tqdm(enumerate(np.linspace(0.05, 0.95, 19))):
    samples = norm.rvs(loc=s, scale=sigma, size=n_samples)
    probs = beta.pdf(samples, a, b) / beta.pdf(s, a, b)
    chosen_samples = samples[np.random.random(n_samples) < probs]
    plt.hist(chosen_samples, density=True, bins=100, color="blue", alpha=0.5)
    plt.plot(x, beta.pdf(x, a, b), color="blue")
    plt.savefig(f"{i}.png")
    plt.close()