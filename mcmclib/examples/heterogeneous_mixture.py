"""
A mixture of two gaussian distributions, one defined on R^1 and another defined on R^2.

14 Dec. 2015
https://github.com/gokererdogan/
"""

import scipy.stats

from mcmclib.hypothesis import *
from mcmclib.parallel_tempering_sampler import *
from mcmclib.proposal import RandomMixtureProposal


class HeterogeneousMixtureHypothesis(Hypothesis):
    def __init__(self, initial_x=None):
        Hypothesis.__init__(self)
        if initial_x is not None:
            self.x = initial_x
            self.k = len(initial_x)
        else:
            self.k = np.random.randint(2) + 1
            self.x = np.random.randn(self.k)

    def _calculate_log_prior(self):
        # p(x) =
        #   1/2 * N(0,1)            if k = 1
        #   1/2 * N(0,1) * N(0,1)   if k = 2
        # px = 0.5 * scipy.stats.norm.pdf(self.x[0])
        px = 0.3 * scipy.stats.norm.pdf(self.x[0])
        if self.k == 2:
            px = px * (0.7 / 0.3) * scipy.stats.norm.pdf(self.x[1])
        return np.log(px)

    def _calculate_log_likelihood(self, data=None):
        # we will sample from the prior
        # likelihood is uniform
        return 0.0

    def copy(self):
        h_copy = HeterogeneousMixtureHypothesis(initial_x=self.x.copy())
        return h_copy

    def __str__(self):
        return str(self.x)

    def __repr__(self):
        return self.__str__()


def switch_space_move(h, params):
    hp = h.copy()
    if hp.k == 1:
        # go up to 2D space
        x2 = np.random.randn()
        hp.x = np.array([hp.x[0], x2])
        hp.k = 2
        q_hp_h = scipy.stats.norm.pdf(x2)
        q_h_hp = 1.0
    else:
        # go down to 1D space
        hp.x = np.array([hp.x[0]])
        hp.k = 1
        q_hp_h = 1.0
        q_h_hp = scipy.stats.norm.pdf(h.x[1])
    return hp, q_hp_h, q_h_hp


def random_walk_move(h, params):
    hp = h.copy()
    step = np.random.randn(hp.k) * np.sqrt(params['MOVE_VARIANCE'])
    hp.x += step
    # p(hp|h) and p(h|hp) are equal
    return hp, 1.0, 1.0


if __name__ == '__main__':
    sample_count = 5000
    h = HeterogeneousMixtureHypothesis()
    proposal = RandomMixtureProposal(moves={'switch_space': switch_space_move, 'random_walk': random_walk_move},
                                     params={'MOVE_VARIANCE': 0.5})
    # Metropolis-Hastings
    import mcmclib.mh_sampler as mh

    mh_sampler = mh.MHSampler(initial_h=h, data=None, proposal=proposal, burn_in=1500, sample_count=sample_count,
                              best_sample_count=sample_count, thinning_period=10, report_period=1000)

    run = mh_sampler.sample()
