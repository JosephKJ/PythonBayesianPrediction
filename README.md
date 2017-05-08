# Bayesian Prediction in Python
This project is based on the OpenBugs Dogs Example data. We model the data from the dogs, to make prediction. The posterior cannot be calculated in closed form as the likelihood is a log linear bernouli distribution and the proir that we take is a gausian. Hence, inorder to approximate the likelihood, we use MCMC to sample some values from a proposal distribution. We use a normal distribution as a proposal distribution. Once ww have sampled values from the proposal distribution, we use these values to do bayesian prediction.

## Output for MCMC

