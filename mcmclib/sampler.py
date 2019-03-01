"""
Inferring 3D Shape from 2D Images

This file contains the abstract Sampler class.

Created on Aug 28, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""


class Sampler(object):
    """
    Abstract Sampler class.
    """

    def __init__(self, initial_h, data, proposal, burn_in=0, sample_count=10, best_sample_count=10, thinning_period=100,
                 report_period=100, verbose=False):
        """
        Initialize sampler.

        initial_h: Initial hypothesis (Hypothesis instance)
        data: observed data. Passed to Hypothesis.likelihood function
        proposal: Proposal class (Proposal instance)
        burn_in: Number of burn-in iterations
        sample_count: Number of samples
        best_sample_count: Number of highest probability samples to keep
        thinning_period: Number of samples to discard before getting the next sample
        report_period: Number of iterations to report sampler status.
        verbose: Reporting verbosity
        """
        self.initial_h = initial_h
        self.data = data
        self.proposal = proposal
        self.burn_in = burn_in
        self.sample_count = sample_count
        self.best_sample_count = best_sample_count
        self.thinning_period = thinning_period
        self.report_period = report_period
        self.verbose = verbose
        self.iter_count = burn_in + (sample_count * thinning_period)

    def sample(self):
        """
        Sample from the chain.
        :return An instance of MCMCRun class
        """
        return NotImplementedError()
