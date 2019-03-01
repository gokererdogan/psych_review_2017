"""
Inferring 3D Shape from 2D Images

This file contains the MHSampler (Metropolis-Hastings) sampler

Created on Aug 28, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

from mcmc_run import *
from sampler import *


class MHSampler(Sampler):
    """
    Metropolis-Hastings sampler class.
    """

    def __init__(self, initial_h, data, proposal, burn_in=100, sample_count=10, best_sample_count=10,
                 thinning_period=100, report_period=200, verbose=False):
        """
        Metropolis-Hastings sampler constructor

        initial_h: Initial hypothesis (Hypothesis instance)
        data: observed data. Passed to Hypothesis.likelihood function
        proposal: Proposal class (Proposal instance)
        burn_in: Number of burn-in iterations
        sample_count: Number of samples
        best_sample_count: Number of highest probability samples to keep
        thinning_period: Number of samples to discard before getting the next sample
        report_period: Number of iterations to report sampler status.
        """
        Sampler.__init__(self, initial_h, data, proposal, burn_in, sample_count, best_sample_count, thinning_period,
                         report_period, verbose)

    def sample(self):
        """
        Sample from the posterior over hypotheses given data using MH algorithm
        """
        h = self.initial_h
        # log posterior of hypothesis
        log_p_h = h.log_prior() + h.log_likelihood(self.data)

        run = MCMCRun(info={"Sampler": "MHSampler"}, log_row_count=self.iter_count,
                      best_sample_count=self.best_sample_count,
                      log_columns=['Iteration', 'IsAccepted', 'LogProbability', 'LogAcceptanceRatio', 'MoveType'])
        accepted_count = 0
        print("MHSampler Start\n")
        for i in range(self.iter_count):

            # propose next state
            move_type, hp, q_hp_h, q_h_hp = self.proposal.propose(h)

            # calculate acceptance ratio
            # note that we already have log p(h), we only log p(hp)
            log_p_hp = hp.log_prior() + hp.log_likelihood(self.data)

            # a(h -> hp)
            log_a_hp_h = log_p_hp + np.log(q_h_hp) - (log_p_h + np.log(q_hp_h))

            is_accepted = 0
            # accept/reject
            if np.log(np.random.rand()) < log_a_hp_h:
                if self.verbose:
                    print(
                        "Iteration {0:d}: p {1:f} {2:f} a {3:f}. Move: {4:s}\n".format(i, log_p_h, log_p_hp, log_a_hp_h,
                                                                                       move_type))
                is_accepted = 1
                accepted_count += 1
                h = hp
                log_p_h = log_p_hp

            run.record_log({'Iteration': i, 'IsAccepted': is_accepted, 'LogProbability': log_p_h,
                            'LogAcceptanceRatio': log_a_hp_h, 'MoveType': move_type})

            if i >= self.burn_in:
                if (i % self.thinning_period) == 0:
                    # NOTE: move_type here is sometimes WRONG. it is not always the move that immediately led to the
                    # move. This should be FIXED at some point.
                    run.add_sample(h, log_p_h, i, move_type)

                run.add_best_sample(h, log_p_h, i, move_type)

            # report sampler state
            if (i % self.report_period) == 0:
                print("Iteration {0:d}, current hypothesis".format(i))
                print("Log posterior probability: {0:e}".format(log_p_h))
                print(h)

        run.finish()
        print("Sampling finished. Acceptance rate: {0:f}\n".format(float(accepted_count) / self.iter_count))
        return run
