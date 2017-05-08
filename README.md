# Bayesian Prediction in Python

This project is based on the OpenBugs Dogs Example data. We model the data from the dogs, to make prediction. The posterior cannot be calculated in closed form as the likelihood is a log linear bernouli distribution and the proir that we take is from a normal distribution. Hence, inorder to approximate the likelihood, we use MCMC to sample some values from a proposal distribution. We use a normal distribution as a proposal distribution. Once we have sampled values from the proposal distribution, we use these values to do bayesian prediction.

## OpenBugs Dogs data

The dogs data is based on Solomon-Wynne experiment with dogs, in which a dog was walked into a iron cage and an electric skock was applied to the cage after raising the bar across the entrance to the cage. Shock was applied only after 10 seconds. The test is to see whether the dog jumps over the bar before 10 seconds. Initially even the smart dogs gets shock, but eventually they will learn from the experiance and jump off before getting a shock. 

30 such dogs are put to 25 such trials. The results are recorded as success (Y = 1 ) if the dog jumps the barrier before the shock occurs, or failure (Y = 0) otherwise. 

## Modelling Likelihood

Consider the event that the dog jumps at trial i and trial j, where j > i. The probability that event j happens is not independent with the event i. In that case, we cannot decompose the likelihood into the product of independent events. To overcome this situation, we consider independent bernouli events with different probailities, where the probability is modeled with respect to the past events. 

The probability of a shock (failure) at trial j can be modeled using this formula:

<a href="https://www.codecogs.com/eqnedit.php?latex=p_{j}&space;=&space;A^{x_{j}}&space;B^{j&space;-&space;x_{j}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_{j}&space;=&space;A^{x_{j}}&space;B^{j&space;-&space;x_{j}}" title="p_{j} = A^{x_{j}} B^{j - x_{j}}" /></a>

where 
* A and B are two varaibles which we will estimate, 
* x<sub>j</sub> is number of success (avoidances) before trial j and
* j - x<sub>j</sub>   = number of previous failures (shocks).
---
p<sub>j</sub> can be expressed as the log linear form as:

<a href="https://www.codecogs.com/eqnedit.php?latex=log&space;\,&space;p_{j}&space;=&space;\alpha&space;(x_{j})&space;&plus;&space;\beta&space;(j&space;-&space;x_{j})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?log&space;\,&space;p_{j}&space;=&space;\alpha&space;(x_{j})&space;&plus;&space;\beta&space;(j&space;-&space;x_{j})" title="log \, p_{j} = \alpha (x_{j}) + \beta (j - x_{j})" /></a>

where 
* alpha and beta are two varaibles which we will estimate
---
The complete likelihood can be calculated as:

<a href="https://www.codecogs.com/eqnedit.php?latex=\prod_{i&space;=&space;1}^{30}\prod_{t&space;=&space;1}^{24}&space;(p_{j}^{y_{j}})(1-p_{j}^{1-y_{j}})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\prod_{i&space;=&space;1}^{30}\prod_{t&space;=&space;1}^{24}&space;(p_{j}^{y_{j}})(1-p_{j}^{1-y_{j}})" title="\prod_{i = 1}^{30}\prod_{t = 1}^{24} (p_{j}^{y_{j}})(1-p_{j}^{1-y_{j}})" /></a>

where 
* p<sub>j</sub> is calculated in the second formula
* y<sub>j</sub> is output of the trial

---

This is how it looks in code:
```python
    def calculate_likelihood(self, alpha, beta):
        p_log = np.zeros((self.n_dogs, self.n_trials), dtype=np.float64)
        p = np.zeros((self.n_dogs, self.n_trials), dtype=np.float64)
        prob = np.zeros((self.n_dogs, self.n_trials), dtype=np.float64)

        p_log = alpha * self.num_success + beta * self.num_failure
        p = np.exp(p_log)

        for d in range(self.n_dogs):
            for t in range(self.n_trials):
                if self.data[d][t] == 0:  # dog did-not jump, hence it got electrocuted
                    prob[d][t] = p[d][t]
                else:
                    prob[d][t] = 1 - p[d][t]

        likelihood = prob.prod()

        return likelihood
```

## Modelling Prior

As we have two variables alpha and beta, we need to put a prior over each of them. 

We assume the prior to be a standard normal. 

## Proposal Distribution

We take a normal distribution with a specific varaince as the proposal distribution. 

## Doing MCMC

As the likelihood is not conjugate to the prior, we will not be able to compute the posterior is closed form. Hence we sample some points from a proposal distribution and accept them based on an acceptance criteria. We believe that the accepted points are coming from the actual posterior which we are trying to model. 


Acceptance Criteria based on Metropolis Hastings:

<a href="https://www.codecogs.com/eqnedit.php?latex=A(z^{*}&space;|&space;z)&space;=&space;\frac{\widetilde{p}(z^{*})\,&space;q(z)}{\widetilde{p}(z)\,&space;q(z^{*})}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A(z^{*}&space;|&space;z)&space;=&space;\frac{\widetilde{p}(z^{*})\,&space;q(z)}{\widetilde{p}(z)\,&space;q(z^{*})}" title="A(z^{*} | z) = \frac{\widetilde{p}(z^{*})\, q(z)}{\widetilde{p}(z)\, q(z^{*})}" /></a>


This is how it looks in code:

```Python

    def mcmc_sampler(self, alpha_init=-1, beta_init=-1, iteration=10000):

        alpha_prev = alpha_init
        beta_prev = beta_init
        n_accepted = 0
        n_rejected = 0
        accepted_alpha = []
        accepted_beta = []
        burn_in = np.ceil(0.1 * iteration)

        for i in range(iteration):
            alpha_new = self.generate_samples()
            beta_new = self.generate_samples()

            # Posterior Calculation
            posterior_prev = self.compute_posterior(alpha_prev, beta_prev)
            posterior_new = self.compute_posterior(alpha_new, beta_new)

            # Proposal distribution pdf value
            proposal_prob_prev = stats.norm.pdf(alpha_prev) * stats.norm.pdf(beta_prev)
            proposal_prob_new = stats.norm.pdf(alpha_new) * stats.norm.pdf(beta_new)

            acceptance_ratio = min(1, (posterior_new * proposal_prob_prev) / (posterior_prev * proposal_prob_new))

            accept = np.random.rand() < acceptance_ratio

            if accept and (i > burn_in):
                alpha_prev = alpha_new
                beta_prev = beta_new

                n_accepted += 1
                accepted_alpha.append(alpha_new)
                accepted_beta.append(beta_new)

            else:
                n_rejected += 1
                accepted_alpha.append(alpha_prev)
                accepted_beta.append(beta_prev)

        self.accepted_alpha = accepted_alpha
        self.accepted_beta = accepted_beta
        
```

## Doing Prediction

So now, we have sampled points from a proposal distribution, which tightly models the intractable posterior distribution. 

Hence, we will use these points to do prediction. 

Bayesian Prediction Formula looks like this:

<a href="https://www.codecogs.com/eqnedit.php?latex=p({\tilde&space;{x}}\mid&space;\mathbf&space;{D}&space;,M&space;)=\int&space;_{\theta&space;}p({\tilde&space;{x}}\mid&space;\theta&space;)p(\theta&space;\mid&space;\mathbf&space;{D}&space;,M&space;)\operatorname&space;{d}&space;\!\theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p({\tilde&space;{x}}\mid&space;\mathbf&space;{D}&space;,M&space;)=\int&space;_{\theta&space;}p({\tilde&space;{x}}\mid&space;\theta&space;)p(\theta&space;\mid&space;\mathbf&space;{D}&space;,M&space;)\operatorname&space;{d}&space;\!\theta" title="p({\tilde {x}}\mid \mathbf {D} ,M )=\int _{\theta }p({\tilde {x}}\mid \theta )p(\theta \mid \mathbf {D} ,M )\operatorname {d} \!\theta" /></a>
