"""
A simple 1D bimodal example for testing ParallelTemperingSampler.

12 Dec. 2015
https://github.com/gokererdogan/
"""

import scipy.stats

from mcmclib.hypothesis import *
from mcmclib.parallel_tempering_sampler import *
from mcmclib.proposal import DeterministicMixtureProposal


class Bimodal1DHypothesis(Hypothesis):
    def __init__(self, initial_x=None):
        Hypothesis.__init__(self)
        self.x = initial_x
        if self.x is None:
            self.x = np.random.randn()

    def _calculate_log_prior(self):
        # assume uniform prior
        return 0.0

    def _calculate_log_likelihood(self, data=None):
        px = 0.6 * scipy.stats.norm.pdf(self.x, -2.0, 0.4) + 0.4 * scipy.stats.norm.pdf(self.x, 2.0, 0.6)
        return np.log(px)

    def copy(self):
        h_copy = Bimodal1DHypothesis(initial_x=self.x)
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
    sample_count = 500
    # parallel tempering
    proposal1 = DeterministicMixtureProposal({'random_walk': random_walk_move}, {'MOVE_VARIANCE': 2.0})
    proposal2 = DeterministicMixtureProposal({'random_walk': random_walk_move}, {'MOVE_VARIANCE': 1.0})
    proposal3 = DeterministicMixtureProposal({'random_walk': random_walk_move}, {'MOVE_VARIANCE': 0.2})
    proposal = [proposal1, proposal2, proposal3]
    h1 = Bimodal1DHypothesis()
    h2 = Bimodal1DHypothesis()
    h3 = Bimodal1DHypothesis()
    h = [h1, h2, h3]
    T = [10.0, 3.0, 1.0]
    sampler = ParallelTemperingSampler(initial_hs=h, data=None, proposals=proposal, temperatures=T,
                                       burn_in=500, sample_count=sample_count, best_sample_count=sample_count,
                                       thinning_period=50, report_period=200)
    run = sampler.sample()
    pt_samples = [s.x for s in run.samples.samples]

    # compare it with Metropolis-Hastings
    import mcmclib.mh_sampler as mh

    mh_sampler = mh.MHSampler(initial_h=h1, data=None, proposal=proposal3, burn_in=1500, sample_count=sample_count,
                              best_sample_count=sample_count, thinning_period=150, report_period=600)
    mh_run = mh_sampler.sample()
    mh_samples = [s.x for s in mh_run.samples.samples]

    # plot results
    x = np.linspace(-5.0, 5.0, 100)
    p1 = scipy.stats.norm.pdf(x, -2.0, 0.4)
    p2 = scipy.stats.norm.pdf(x, 2.0, 0.6)
    p = 0.6 * p1 + 0.4 * p2

    import matplotlib.pyplot as plt

    plt.figure()
    plt.hold(True)
    plt.hist(mh_samples, bins=20, normed=True)
    plt.plot(x, p)
    plt.title("Samples with Metropolis-Hastings")

    plt.figure()
    plt.hold(True)
    plt.hist(pt_samples, bins=20, normed=True)
    plt.plot(x, p)
    plt.title("Samples with Parallel Tempering")
