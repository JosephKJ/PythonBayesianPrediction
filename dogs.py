import numpy as np
import scipy.stats as stats


class Dogs:
    data = []
    y = []
    n_dogs = 0
    n_trials = 0
    num_success = []
    num_failure = []
    accepted_alpha = []
    accepted_beta = []

    def __init__(self, data):
        self.data = data
        self.n_dogs, self.n_trials = data.shape
        self.flip_data()
        self.calculate_number_of_success_failure()

    def show_data(self):
        print self.data

    def flip_data(self):
        self.y = 1 - self.data

    def calculate_number_of_success_failure(self):
        self.num_success = np.zeros((self.n_dogs, self.n_trials), dtype=np.int32) # No shock
        self.num_failure = np.zeros((self.n_dogs, self.n_trials), dtype=np.int32)
        for d in range(self.n_dogs):
            self.num_success[d,0] = 0
            self.num_failure[d,0] = 0
            for t in range(1, self.n_trials):
                for i in range(0, t):
                    self.num_success[d, t] = self.num_success[d, t] + self.data[d, i]
                self.num_failure[d, t] = t - self.num_success[d, t]

    def calculate_likelihood(self, alpha, beta):
        p_log = np.zeros((self.n_dogs, self.n_trials), dtype=np.float64)
        p = np.zeros((self.n_dogs, self.n_trials), dtype=np.float64)

        p_log = alpha * self.num_success + beta * self.num_failure
        p = np.exp(p_log)

        prob = np.zeros((self.n_dogs, self.n_trials), dtype=np.float64)

        # for d in range(self.n_dogs):
        #     for t in range(self.n_trials):
        #         prob[d][t] = stats.bernoulli(p[d][t]).pmf(self.y[d][t])
        #
        # likelihood = prob.prod()

        for d in range(self.n_dogs):
            for t in range(self.n_trials):
                if self.data[d][t] == 0:  # dog did-not jump, hence it got electrocuted
                    prob[d][t] = p[d][t]
                else:
                    prob[d][t] = 1 - p[d][t]

        likelihood = prob.prod()

        return likelihood

    def compute_posterior(self, alpha, beta, prior=None):
        likelihood = self.calculate_likelihood(alpha, beta)

        if prior:
            posterior = likelihood * prior
        else:
            alpha_prior = stats.norm.pdf(alpha)
            beta_prior = stats.norm.pdf(beta)
            posterior = likelihood * alpha_prior * beta_prior

        return posterior

    def generate_samples(self):
        while True:
            val = stats.norm.rvs(scale=.36)
            if val < -0.00001:
                return val

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
            if i > burn_in:
                if accept:
                    alpha_prev = alpha_new
                    beta_prev = beta_new

                    n_accepted += 1
                    accepted_alpha.append(alpha_new)
                    accepted_beta.append(beta_new)

                else:
                    n_rejected += 1
                    accepted_alpha.append(alpha_prev)
                    accepted_beta.append(beta_prev)

        print "\nStatistics of alpha and beta"
        print "----------------------------"
        print "Number of accepted samples: %d " % n_accepted
        print "Number of rejected samples: %d " % n_rejected
        print "Mean of alpha values: %f" % (np.mean(accepted_alpha))
        print "Mean of beta values: %f" % (np.mean(accepted_beta))

        self.accepted_alpha = accepted_alpha
        self.accepted_beta = accepted_beta

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

        print "\nPrediction"
        print "----------"
        print "Number of instances where the dog jumps off: %d" % num_success
        print "Number of instances where the dog gets shock: %d" % num_failure
        print "Prediction: "
        print prediction
        print "Probability values:"
        print prob_values
        print "\nLegend:\n'True' indicates avoidance of shock and 'False' indicates event of getting shock."


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
d.mcmc_sampler(-1, -1, 10000)
d.predict()
