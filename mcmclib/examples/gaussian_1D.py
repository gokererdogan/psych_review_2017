"""
A simple MH sampler example for sampling from a 1D Gaussian distribution.

2 Dec. 2015
https://github.com/gokererdogan/
"""

from mcmclib.hypothesis import *
from mcmclib.mh_sampler import *
from mcmclib.proposal import DeterministicMixtureProposal


class Gaussian1DHypothesis(Hypothesis):
    def __init__(self, initial_x=None, variance=1.0):
        Hypothesis.__init__(self)
        self.x = initial_x
        if self.x is None:
            self.x = np.random.randn()
        self.variance = variance

    def _calculate_log_prior(self):
        # assume uniform prior
        return 0.0

    def _calculate_log_likelihood(self, data=None):
        return -(np.sum(np.square(self.x - data)) / (2 * self.variance))

    def copy(self):
        h_copy = Gaussian1DHypothesis(initial_x=self.x, variance=self.variance)
        return h_copy

    def __str__(self):
        return str(self.x)

    def __repr__(self):
        return self.__str__()


def random_walk_move(h, params):
    hp = h.copy()
    step = np.random.randn() * np.sqrt(params['MOVE_VARIANCE'])
    hp.x += step
    # p(hp|h) and p(h|hp) are equal
    return hp, 1.0, 1.0


if __name__ == '__main__':
    mu = 2.0
    data = np.random.randn(10000) + mu
    proposal = DeterministicMixtureProposal({'random_walk': random_walk_move}, {'MOVE_VARIANCE': 0.001})
    h = Gaussian1DHypothesis()
    sampler = MHSampler(initial_h=h, data=data, proposal=proposal, burn_in=500, sample_count=10, best_sample_count=10,
                        thinning_period=500, report_period=500)
    run = sampler.sample()
