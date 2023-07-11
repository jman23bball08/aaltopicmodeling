import os
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import torch

assert pyro.__version__.startswith('1.8.5')
# Enable smoke test - run the notebook cells on CI.
smoke_test = 'CI' in os.environ

def model(counts):
    theta = pyro.sample('theta', dist.Dirichlet(torch.ones(6)))
    total_count = int(counts.sum())
    pyro.sample('counts', dist.Multinomial(total_count, theta), obs=counts)

data = torch.tensor([5, 4, 2, 5, 6, 5, 3, 3, 1, 5, 5, 3, 5, 3, 5, \
                     3, 5, 5, 3, 5, 5, 3, 1, 5, 3, 3, 6, 5, 5, 6])
counts = torch.unique(data, return_counts=True)[1].float()

nuts_kernel = NUTS(model)
num_samples, warmup_steps = (1000, 200) if not smoke_test else (10, 10)
mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
mcmc.run(counts)
hmc_samples = {k: v.detach().cpu().numpy()
               for k, v in mcmc.get_samples().items()}

means = hmc_samples['theta'].mean(axis=0)
stds = hmc_samples['theta'].std(axis=0)
print('Inferred dice probabilities from the data (68% confidence intervals):')
for i in range(6):
    print('%d: %.2f ± %.2f' % (i + 1, means[i], stds[i]))