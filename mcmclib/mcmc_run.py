"""
Inferring 3D Shape from 2D Images

This file contains the MCMCRun and related classes. These classes store the results of an MCMC run.

Created on Aug 28, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import cPickle
import time

import numpy as np
import pandas as pd
import pandasql as psql

BEST_SAMPLES_LIST_SIZE = 20


class MCMCRun(object):
    """
    MCMCRun class holds information, e.g., probability, acceptance rate,
     samples, and best samples, related to a run of a MCMC chain.
    """

    def __init__(self, info, log_row_count, log_columns, best_sample_count=BEST_SAMPLES_LIST_SIZE):
        self.info = info
        self.start_time = time.strftime("%Y.%m.%d %H:%M:%S")
        self.end_time = ""
        self.samples = SampleSet()
        self.best_samples = BestSampleSet(best_sample_count)
        self.log_row_count = log_row_count
        self.run_log = pd.DataFrame(index=np.arange(0, self.log_row_count), dtype=np.float, columns=log_columns)
        self.last_log_row_index = 0

    def record_log(self, row):
        """Record one row into log

        Args:
            row (dict): A dictionary of information to record
        """
        if self.last_log_row_index >= self.log_row_count:
            raise IndexError("Log full. Cannot add new row.")

        self.run_log.iloc[self.last_log_row_index] = pd.Series(row)
        self.last_log_row_index += 1

    def add_sample(self, s, log_prob, iter_no, info):
        self.samples.add(s, log_prob, iter_no, info)

    def add_best_sample(self, s, log_prob, iter_no, info):
        self.best_samples.add(s, log_prob, iter_no, info)

    def finish(self):
        self.end_time = time.strftime("%Y.%m.%d %H:%M:%S")

    def plot_probs(self):
        self.run_log.plot('Iteration', 'LogProbability')

    def plot_acceptance_rate(self, window_size=100):
        # calculate moving average
        pd.rolling_mean(self.run_log.IsAccepted, window=window_size).plot()

    def acceptance_rate_by_move(self):
        df = self.run_log
        return psql.sqldf("select MoveType, AVG(IsAccepted) as AcceptanceRate from df group by MoveType", env=locals())

    def save(self, filename):
        cPickle.dump(obj=self, file=open(filename, 'wb'), protocol=2)

    @staticmethod
    def load(filename):
        return cPickle.load(open(filename, 'r'))


class SampleSet:
    """
    SampleSet class implements a simple list of samples.
    Each sample consists of the hypothesis, its log posterior
    probability and some info associated with it.
    """

    def __init__(self):
        self.samples = []
        self.log_probs = []
        self.infos = []
        self.iters = []

    def add(self, s, log_prob, iter_no, info):
        self.samples.append(s)
        self.log_probs.append(log_prob)
        self.infos.append(info)
        self.iters.append(iter_no)

    def pop(self, i):
        if i < len(self.samples):
            s = self.samples.pop(i)
            log_prob = self.log_probs.pop(i)
            iter_no = self.iters.pop(i)
            info = self.infos.pop(i)
            return s, log_prob, iter_no, info
        return None

    def __getitem__(self, item):
        if item < len(self.samples):
            s = self.samples[item]
            log_prob = self.log_probs[item]
            iter_no = self.iters[item]
            info = self.infos[item]
            return s, log_prob, iter_no, info
        raise KeyError("Index exceeds number of samples in sample set.")


class BestSampleSet(SampleSet):
    """
    BestSampleSet class implements a list of samples intended to
    keep the best samples (in terms of probability) so far in a
    chain. We add a sample to the set if it has higher probability
    than at least one of the samples in the set.
    """

    def __init__(self, capacity):
        SampleSet.__init__(self)
        self.capacity = capacity

    def add(self, s, log_prob, iter_no, info):
        if len(self.samples) < self.capacity:
            if s not in self.samples:
                SampleSet.add(self, s, log_prob, iter_no, info)
        elif log_prob > np.min(self.log_probs):
            if s not in self.samples:
                min_i = np.argmin(self.log_probs)
                self.pop(min_i)
                SampleSet.add(self, s, log_prob, iter_no, info)
