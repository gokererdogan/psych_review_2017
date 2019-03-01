"""
mcmclib

Unit tests for Sampler and MHSampler classes.

Created on Dec 2, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import unittest

from mcmclib.examples.gaussian_1D import *


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


class MHSamplerTest(unittest.TestCase):
    def test_dummy_hypothesis_sampler(self):
        h = DummyHypothesis(magic_key=1234)
        p = DeterministicMixtureProposal(moves={'dummy_move': dummy_move}, params=None)
        self.s = MHSampler(initial_h=h, proposal=p, data=None, burn_in=0, sample_count=2, best_sample_count=2,
                           thinning_period=1)

        # chain should run for 2 iterations
        self.assertEqual(self.s.iter_count, 2)
        run = self.s.sample()
        # all moves should be accepted
        self.assertEqual(np.sum(run.run_log.IsAccepted), 2)
        # log acceptance ratios should be 0.0
        self.assertTrue(np.all(run.run_log.LogAcceptanceRatio == 0.0))
        # all log probabilities should be 0.0
        self.assertTrue(np.all(run.run_log.LogProbability == 0.0))
        # all move types should be dummy_move
        self.assertTrue(np.all(run.run_log.MoveType == 'dummy_move'))
        # samples should have magic keys 1235, 1236 (remember: initial sample is not recorded)
        self.assertEqual(run.samples.samples[0].magic_key, 1235)
        self.assertEqual(run.samples.samples[1].magic_key, 1236)
        self.assertEqual(run.best_samples.samples[0].magic_key, 1235)
        self.assertEqual(run.best_samples.samples[1].magic_key, 1236)

    def test_dummy_hypothesis2_sampler(self):
        h = DummyHypothesis2(p=1.0)
        p = DeterministicMixtureProposal(moves={'dummy2_move': dummy2_move}, params=None)
        s = MHSampler(initial_h=h, proposal=p, data=None, burn_in=0, sample_count=2, best_sample_count=2,
                      thinning_period=1)

        # chain should run for 2 iterations
        self.assertEqual(s.iter_count, 2)
        run = s.sample()
        print(run.run_log)
        # first acceptance rate: p(h) = 1.0, p(h') = 2.0, q(hp|h) = 0.5, q(h|hp) = 1.0.
        # therefore a(h -> h') = (2.0 * 1.0) / (1.0 * 0.5) = 4.0
        self.assertAlmostEqual(run.run_log.LogAcceptanceRatio[0], np.log(4.0))
        # second acceptance rate: p(h) = 2.0, p(h') = 3.0, q(hp|h) = 0.5, q(h|hp) = 1.0.
        # therefore a(h -> h') = (3.0 * 1.0) / (2.0 * 0.5) = 3.0
        self.assertAlmostEqual(run.run_log.LogAcceptanceRatio[1], np.log(3.0))
        # log probabilities
        self.assertAlmostEqual(run.run_log.LogProbability[0], np.log(2.0))
        self.assertAlmostEqual(run.samples.samples[0].log_prior() + run.samples.samples[0].log_likelihood(),
                               np.log(2.0))
        self.assertAlmostEqual(run.run_log.LogProbability[1], np.log(3.0))
        self.assertAlmostEqual(run.samples.samples[1].log_prior() + run.samples.samples[1].log_likelihood(),
                               np.log(3.0))
        # all move types should be dummy2_move
        self.assertTrue(np.all(run.run_log.MoveType == 'dummy2_move'))

    def test_gaussian_1D_sampler(self):
        mu = 2.0
        data = np.random.randn(10000) + mu
        proposal = DeterministicMixtureProposal({'random_walk': random_walk_move}, {'MOVE_VARIANCE': 0.001})
        h = Gaussian1DHypothesis()
        sampler = MHSampler(initial_h=h, data=data, proposal=proposal, burn_in=500, sample_count=10,
                            best_sample_count=10, thinning_period=500, report_period=500)
        run = sampler.sample()
        # we would expect the samples be quite close to 2.0
        x = np.array([s.x for s in run.samples.samples])
        self.assertTrue(np.all(np.abs(x - mu) < 0.05))
        x = np.array([s.x for s in run.best_samples.samples])
        self.assertTrue(np.all(np.abs(x - mu) < 0.05))
