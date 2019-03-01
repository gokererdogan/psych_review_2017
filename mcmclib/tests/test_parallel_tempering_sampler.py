"""
mcmclib

Unit tests for ParallelTemperingSampler class.

Created on Dec 12, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import unittest

from mcmclib.examples.gaussian_1D import *
from mcmclib.parallel_tempering_sampler import *


class DummyHypothesis(Hypothesis):
    def __init__(self, magic_key):
        Hypothesis.__init__(self)
        self.magic_key = magic_key

    def _calculate_log_prior(self):
        return 0.0

    def _calculate_log_likelihood(self, data=None):
        return 0.0

    def copy(self):
        return DummyHypothesis(magic_key=self.magic_key + 1)


def dummy_move(h, params):
    hp = h.copy()
    return hp, 1.0, 1.0


class DummyHypothesis2(Hypothesis):
    def __init__(self, p):
        Hypothesis.__init__(self)
        self.p = p

    def _calculate_log_prior(self):
        return np.log(self.p)

    def _calculate_log_likelihood(self, data=None):
        return 0.0

    def copy(self):
        return DummyHypothesis2(p=self.p)


def dummy2_move(h, params):
    hp = h.copy()
    hp.p = h.p + 1.0
    return hp, 0.5, 1.0


class ParallelTemperingSamplerTest(unittest.TestCase):
    def test_dummy_hypothesis_sampler(self):
        h1 = DummyHypothesis(magic_key=1234)
        h2 = DummyHypothesis(magic_key=5678)
        h = [h1, h2]
        p1 = DeterministicMixtureProposal(moves={'dummy_move': dummy_move}, params=None)
        p2 = DeterministicMixtureProposal(moves={'dummy_move': dummy_move}, params=None)
        p = [p1, p2]
        t = [-10.0, 100.0]
        # temperature has to be positive
        self.assertRaises(ValueError, ParallelTemperingSampler, h, None, p, t)
        # initial_hs, proposals and temperatures should have the same length
        self.assertRaises(ValueError, ParallelTemperingSampler, [h1], None, p, t)
        self.assertRaises(ValueError, ParallelTemperingSampler, h, None, [p1], t)
        self.assertRaises(ValueError, ParallelTemperingSampler, h, None, p, [100.00])
        # sampling_chain cannot be omitted if there are no chains sampling at temperature 1.0
        self.assertRaises(ValueError, ParallelTemperingSampler, h, None, p, [100.00, 10.0])

        s = ParallelTemperingSampler(initial_hs=h, proposals=p, data=None, temperatures=[10.0, 1.0],
                                     burn_in=0, sample_count=2, best_sample_count=2, thinning_period=1)
        self.assertEqual(s.chain_count, 2)
        self.assertEqual(s.sampling_chain, 1)

        run = s.sample()
        # there should be 6 lines in run_log (at each iteration, we update 2 chain + exchange move)
        self.assertEqual(run.run_log.shape[0], 6)
        # all moves should be accepted
        self.assertEqual(np.sum(run.run_log.IsAccepted), 6)
        # log acceptance ratios should be 0.0
        self.assertTrue(np.all(run.run_log.LogAcceptanceRatio == 0.0))
        # all log probabilities should be 0.0
        self.assertTrue(np.all(run.run_log.LogProbability == 0.0))
        # 4 of 6 moves should be dummy_move
        self.assertEqual(np.sum(run.run_log.MoveType == 'dummy_move'), 4)
        # rest should be exchange
        self.assertEqual(np.sum(run.run_log.MoveType == 'exchange'), 2)
        # samples should have magic keys 1235, 5680 (remember: initial sample is not recorded)
        self.assertEqual(run.samples.samples[0].magic_key, 1235)
        self.assertEqual(run.samples.samples[1].magic_key, 5680)
        self.assertEqual(run.best_samples.samples[0].magic_key, 1235)
        self.assertEqual(run.best_samples.samples[1].magic_key, 5680)

    def test_dummy_hypothesis2_sampler(self):
        h1 = DummyHypothesis2(p=1.0)
        h2 = DummyHypothesis2(p=2.0)
        h = [h1, h2]
        p1 = DeterministicMixtureProposal(moves={'dummy2_move': dummy2_move}, params=None)
        p2 = DeterministicMixtureProposal(moves={'dummy2_move': dummy2_move}, params=None)
        p = [p1, p2]
        t = [2.0, 1.0]
        s = ParallelTemperingSampler(initial_hs=h, proposals=p, data=None, temperatures=t,
                                     burn_in=0, sample_count=1, best_sample_count=1, thinning_period=1)
        self.assertEqual(s.chain_count, 2)
        self.assertEqual(s.sampling_chain, 1)

        run = s.sample()
        # there should be 3 lines in run_log (at each iteration, we update 2 chain + exchange move)
        self.assertEqual(run.run_log.shape[0], 3)
        # first iteration
        # first chain (T=2.0), within chain update. p(h) = (1.0)^(1/2), p(hp) = 2.0^(1/2), q(hp|h) = 0.5, q(h|hp) = 1.0
        #   a(h->hp) = [2.0^(1/2) * 1.0] / [1.0^(1/2) * 0.5] = 2*sqrt(2)
        self.assertAlmostEqual(run.run_log.LogAcceptanceRatio[0], np.log(np.sqrt(2.0) * 2.0))
        # second chain (T=1.0), within chain update. p(h) = 2.0, p(hp) = 3.0, q(hp|h) = 1.0, q(h|hp) = 0.5
        #   a(h->hp) = [3.0 * 1.0] / [2.0 * 0.5] = 3.0
        self.assertAlmostEqual(run.run_log.LogAcceptanceRatio[1], np.log(3.0))
        # exchange move, p(h1) = 2.0, p(h2) = 3.0, T1 = 2.0, T2 = 1.0
        #   a(c1<->c2) = [3.0^(1/2) * 2.0^(1/1)] / [3.0^(1/1) * 2.0^(1/2)] = (3/2)^(-1/2)
        self.assertAlmostEqual(run.run_log.LogAcceptanceRatio[2], np.log(1 / np.sqrt(1.5)))

    def test_gaussian_1D_sampler(self):
        mu = 2.0
        data = np.random.randn(10000) + mu
        proposal1 = DeterministicMixtureProposal({'random_walk': random_walk_move}, {'MOVE_VARIANCE': 0.01})
        proposal2 = DeterministicMixtureProposal({'random_walk': random_walk_move}, {'MOVE_VARIANCE': 0.001})
        proposal = [proposal1, proposal2]
        h1 = Gaussian1DHypothesis()
        h2 = Gaussian1DHypothesis()
        h = [h1, h2]
        T = [10.0, 1.0]
        sampler = ParallelTemperingSampler(initial_hs=h, data=data, proposals=proposal, temperatures=T,
                                           burn_in=500, sample_count=10, best_sample_count=10, thinning_period=500,
                                           report_period=500)
        run = sampler.sample()
        # we would expect the samples be quite close to 2.0
        x = np.array([s.x for s in run.samples.samples])
        self.assertTrue(np.all(np.abs(x - mu) < 0.05))
        x = np.array([s.x for s in run.best_samples.samples])
        self.assertTrue(np.all(np.abs(x - mu) < 0.05))
