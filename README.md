# Bayesian Prediction in Python

This project is based on the OpenBugs Dogs Example data. We model the data from the dogs, to make prediction. The posterior cannot be calculated in closed form as the likelihood is a log linear bernouli distribution and the proir that we take is from a normal distribution. Hence, inorder to approximate the likelihood, we use MCMC to sample some values from a proposal distribution. We use a normal distribution as a proposal distribution. Once we have sampled values from the proposal distribution, we use these values to do bayesian prediction.

### OpenBugs Dogs data

The dogs data is based on Solomon-Wynne experiment with dogs, in which a dog was walked into a iron cage and an electric skock was applied to the cage after raising the bar across the entrance to the cage. Shock was applied only after 10 seconds. The test is to see whether the dog jumps the bar before 10 seconds. Initially even the smart dogs gets shock, but eventually they will learn from the experiance and jump off before getting a shock. 

30 such dogs are put to 25 such trials. The results are recorded as success (Y = 1 ) if the dog jumps the barrier before the shock occurs, or failure (Y = 0) otherwise. 

### Modelling Likelihood
