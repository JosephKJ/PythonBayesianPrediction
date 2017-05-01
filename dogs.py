import numpy as np
import scipy.stats as stats

class Dogs:
    data = []
    y = []
    n_dogs = 0
    n_trials = 0

    def __init__(self, data):
        self.data = data
        self.n_dogs, self.n_trials = data.shape

    def show_data(self):
        print self.data

    def flip_data(self):
        self.y = 1 - self.data

    def calculate_likelihood(self, alpha, beta):
        num_success = np.zeros((self.n_dogs, self.n_trials), dtype=np.int32) # No shock
        num_failure = np.zeros((self.n_dogs, self.n_trials), dtype=np.int32)
        for d in range(self.n_dogs):
            num_success[d,0] = 0
            num_failure[d,0] = 0
            for t in range(1, self.n_trials):
                for i in range(0, t):
                    num_success[d, t] = num_success[d, t] + self.data[d, i]
                num_failure[d, t] = t - num_success[d, t]

        p_log = np.zeros((self.n_dogs, self.n_trials), dtype=np.float64)
        p = np.zeros((self.n_dogs, self.n_trials), dtype=np.float64)

        for d in range(self.n_dogs):
            for t in range(self.n_trials):
                p_log[d][t] = alpha * num_success[d][t] + beta * num_failure[d][t]

        p = np.exp(p_log)
        prob = np.zeros((self.n_dogs, self.n_trials), dtype=np.float64)

        for d in range(self.n_dogs):
            for t in range(self.n_trials):
                prob[d][t] = stats.bernoulli(p[d][t]).pmf(1) # self.y[d][t]

        likelihood = prob.prod()

        return likelihood

    def mcmc_sampler(self, alpha_init, beta_init, interation=5):

        alpha_prev = alpha_init
        beta_prev = beta_init
        n_accepted = 0
        accepted_alpha = [alpha_init]
        accepted_beta = [beta_init]


        for _ in range(interation):
            # loc specifies the mean, scale is the standard deviation
            alpha_new = stats.expon.rvs(scale=.0005)
            beta_new = stats.expon.rvs(scale=.0005)

            # Posterior Calculation
            likelihood_prev = self.calculate_likelihood(alpha_prev, beta_prev)
            likelihood_new = self.calculate_likelihood(alpha_new, beta_new)

            alpha_prior_prev = stats.norm.pdf(alpha_prev)
            beta_prior_prev = stats.norm.pdf(beta_prev)

            alpha_prior_new = stats.norm.pdf(alpha_new)
            beta_prior_new = stats.norm.pdf(beta_new)

            posterior_prev = likelihood_prev * alpha_prior_prev * beta_prior_prev
            posterior_new = likelihood_new * alpha_prior_new * beta_prior_new


            # Proposal distribution pdf value
            proposal_prob_prev = stats.expon.pdf(alpha_prev) * stats.expon.pdf(beta_prev)
            proposal_prob_new = stats.expon.pdf(alpha_new) * stats.expon.pdf(beta_new)

            acceptance_ratio = (posterior_new * proposal_prob_new) / (posterior_prev * proposal_prob_prev)

            accept = np.random.rand() < acceptance_ratio

            if accept:
                alpha_prev = alpha_new
                beta_prev = beta_prev

                n_accepted += 1
                accepted_alpha.append(alpha_new)
                accepted_beta.append(beta_new)
                print n_accepted

        return (accepted_alpha, accepted_beta)



data = (0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0,
           0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1,
           0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1,
           0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1,
           0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
n_dogs = 30
n_trial = 25
data = np.array(data).reshape(n_dogs, n_trial)

d = Dogs(data)
d.flip_data()
# d.calculate_likelihood(-0.00001, -0.00001)
print d.mcmc_sampler(-0.00001, -0.00001, 10000)
