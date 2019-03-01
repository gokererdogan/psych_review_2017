"""
mcmclib

Unit tests for Hypothesis class.
There isn't much to test here. We test the prior and likelihood caching mechanisms.

Created on Dec 2, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import unittest

import numpy as np

from mcmclib.hypothesis import *


class DummyHypothesis(Hypothesis):
    def _calculate_log_prior(self):
        return np.random.rand()

    def _calculate_log_likelihood(self, data=None):
        return np.random.rand()


class HypothesisTest(unittest.TestCase):
    def setUp(self):
        self.h = DummyHypothesis()

    def tearDown(self):
        self.h = None

    def test_prior_caching(self):
        val1 = self.h.log_prior()
        val2 = self.h.log_prior()
        self.assertAlmostEqual(val1, val2)

    def test_likelihood_caching(self):
        val1 = self.h.log_likelihood()
        val2 = self.h.log_likelihood()
        self.assertAlmostEqual(val1, val2)
