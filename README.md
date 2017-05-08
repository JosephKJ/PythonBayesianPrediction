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

Bayesian Prediction Formula is:

<a href="https://www.codecogs.com/eqnedit.php?latex=p({\tilde&space;{x}}\mid&space;\mathbf&space;{D}&space;,M&space;)=\int&space;_{\theta&space;}p({\tilde&space;{x}}\mid&space;\theta&space;)p(\theta&space;\mid&space;\mathbf&space;{D}&space;,M&space;)\operatorname&space;{d}&space;\!\theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p({\tilde&space;{x}}\mid&space;\mathbf&space;{D}&space;,M&space;)=\int&space;_{\theta&space;}p({\tilde&space;{x}}\mid&space;\theta&space;)p(\theta&space;\mid&space;\mathbf&space;{D}&space;,M&space;)\operatorname&space;{d}&space;\!\theta" title="p({\tilde {x}}\mid \mathbf {D} ,M )=\int _{\theta }p({\tilde {x}}\mid \theta )p(\theta \mid \mathbf {D} ,M )\operatorname {d} \!\theta" /></a>

Using Monte Carlo method, we can approximate this integral by averaging the function over the variables, drawn from the probability distribution. 

<a href="https://www.codecogs.com/eqnedit.php?latex=p({\tilde&space;{x}}\mid&space;\mathbf&space;{D}&space;,M&space;)=\frac{1}{L}\sum_{i&space;=&space;1}^{L}&space;p({\tilde&space;{x}}\mid&space;\alpha_{i},&space;\beta_{i}&space;)\,&space;where&space;\,&space;\alpha_{i},&space;\beta_{i}&space;\,&space;are&space;\,drawn&space;\,from&space;\,&space;p(\theta&space;\mid&space;\mathbf&space;{D}&space;,M&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p({\tilde&space;{x}}\mid&space;\mathbf&space;{D}&space;,M&space;)=\frac{1}{L}\sum_{i&space;=&space;1}^{L}&space;p({\tilde&space;{x}}\mid&space;\alpha_{i},&space;\beta_{i}&space;)\,&space;where&space;\,&space;\alpha_{i},&space;\beta_{i}&space;\,&space;are&space;\,drawn&space;\,from&space;\,&space;p(\theta&space;\mid&space;\mathbf&space;{D}&space;,M&space;)" title="p({\tilde {x}}\mid \mathbf {D} ,M )=\frac{1}{L}\sum_{i = 1}^{L} p({\tilde {x}}\mid \alpha_{i}, \beta_{i} )\, where \, \alpha_{i}, \beta_{i} \, are \,drawn \,from \, p(\theta \mid \mathbf {D} ,M )" /></a>

This is how it looks in code:

```python

    def predict(self):

        num_success = 0  # number of success (avoidences) before trial j
        num_failure = 0  # number of previous failures (shocks)
        prediction = []
        prob_values = []
        for _ in range(0,25):
            pred = 0
            for i in range (0, len(self.accepted_alpha)):
                log_p = self.accepted_alpha[i] * num_success + self.accepted_beta[i] * num_failure
                p = np.exp(log_p)
                pred = pred + p

            pred = pred / len(self.accepted_alpha)

            if pred > 0.5:
                num_failure += 1
            else:
                num_success += 1

            prediction.append(pred < 0.5)
            prob_values.append(pred)
            
```

## Results

As we have trained a model, we can put to test. 

### Test 1: 

Here we test the case in which a dog is put to the cage the very fist time. In this situation, the dog has never had any experiance on whats going to happen. We can simulate this by setting the values of num_success and num_failure to zero. 

#### Output:

```
/Users/josephkj/anaconda2/bin/python -W ignore /Users/josephkj/PycharmProjects/BayesianProject/dogs.py

Statistics of alpha and beta
----------------------------
Number of accepted samples: 117 
Number of rejected samples: 9883 
Mean of alpha values: -0.318505
Mean of beta values: -0.172197

Prediction
----------
Number of instances where the dog jumps off: 17
Number of instances where the dog gets shock: 8
Prediction: 
[False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
Probability values:
[1.0, 0.86755959855601417, 0.78062268346274488, 0.71342633179041071, 0.6562475532239872, 0.60528633691800149, 0.55894105308408648, 0.51643613941242172, 0.47731553854955466, 0.37391973759195563, 0.29315488484744912, 0.2300164830560234, 0.18061920716148455, 0.14194293943388461, 0.11163783448631691, 0.087874023466147352, 0.069225440086758239, 0.054579877346793378, 0.043069272135382974, 0.034015618351672383, 0.026888970656175393, 0.021274811234490786, 0.016848673499389413, 0.013356394775123596, 0.010598738441596915]

Legend:
'True' indicates avoidance of shock and 'False' indicates event of getting shock.

Process finished with exit code 0

```

#### Inference

The most interesting part of this result is that the probability with which the dog gets the shock in the first trial is predicted to be 1.0 This completely justifies the training data

---

### Test 2: 

Here we test the case in which the dog already had a shock. We can simulate this by setting the value of num_success to 0 and num_failure to 1. 

#### Output:

```
/Users/josephkj/anaconda2/bin/python -W ignore /Users/josephkj/PycharmProjects/BayesianProject/dogs.py

Statistics of alpha and beta
----------------------------
Number of accepted samples: 87 
Number of rejected samples: 9913 
Mean of alpha values: -0.320770
Mean of beta values: -0.169978

Prediction
----------
Number of instances where the dog jumps off: 18
Number of instances where the dog gets shock: 8
Prediction: 
[False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
Probability values:
[0.86953561341847818, 0.78416712955414736, 0.71821746090898098, 0.66202270817755893, 0.61182700912032961, 0.56606367735568142, 0.52398584378391921, 0.48516153840991272, 0.37945022316779098, 0.29687966202362931, 0.23235869016197402, 0.18192334026074511, 0.14248475943944486, 0.11163465277671025, 0.087494503559732817, 0.068598466725721147, 0.053802304523344183, 0.042212519151578233, 0.033131173047105465, 0.026012905423824419, 0.020431436567500332, 0.016053456381653636, 0.012618262358184381, 0.0099218757282912608, 0.0078046468442014077]

Legend:
'True' indicates avoidance of shock and 'False' indicates event of getting shock.

Process finished with exit code 0

```

#### Inference

As we expect, the probability with which the dig gets a shock will decrease, as he had some experiance. The probability has come down to 0.86953561341847818 for the first try.

---

### Test 3: 

Here we test the case in which the dog already started jumping even before getting the shock. This case it like the dog is too smart that it just senses that something bad is going to happen and jumps out proactively. We can simulate this by setting the value of num_success to 1 and num_failure to 0. 

#### Output:

```
/Users/josephkj/anaconda2/bin/python -W ignore /Users/josephkj/PycharmProjects/BayesianProject/dogs.py

Statistics of alpha and beta
----------------------------
Number of accepted samples: 114 
Number of rejected samples: 9886 
Mean of alpha values: -0.315095
Mean of beta values: -0.174344

Prediction
----------
Number of instances where the dog jumps off: 21
Number of instances where the dog gets shock: 5
Prediction: 
[False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
Probability values:
[0.74568564602059784, 0.66623189521141635, 0.60604220333263648, 0.55543712855856353, 0.5106494703692056, 0.47010464931432633, 0.37013412373078325, 0.29162674378683723, 0.22991114745722357, 0.18135811709297292, 0.14313543717071431, 0.11302742993960946, 0.089298168516623339, 0.070586217907712911, 0.055823006886168834, 0.04416926632863713, 0.034965421441929267, 0.027692818954096862, 0.021943392063869062, 0.017395907829440702, 0.01379735569937688, 0.010948355091954763, 0.0086917072147943471, 0.0069034084075926859, 0.0054855917584077157]

Legend:
'True' indicates avoidance of shock and 'False' indicates event of getting shock.

Process finished with exit code 0

```

#### Inference

As we expect, the probability with which the dig gets a shock will decrease further, as he is too proactive. The probability has come down to 0.74568564602059784 for the first try.
