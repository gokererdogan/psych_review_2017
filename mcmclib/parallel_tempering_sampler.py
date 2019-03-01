"""
Inferring 3D Shape from 2D Images

This file contains the Parallel Tempering (PT) sampler.

Created on Dec 11, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

from mcmc_run import *
from sampler import *


class ParallelTemperingSampler(Sampler):
    """Parallel Tempering (PT) sampler class.
    """

    def __init__(self, initial_hs, data, proposals, temperatures, sampling_chain=None, burn_in=100, sample_count=10,
                 best_sample_count=10, thinning_period=100, report_period=200, verbose=False):
        """Parallel Tempering (PT) sampler constructor

        Args:
            initial_h (list): List of initial hypotheses, one for each chain (Hypothesis instance)
            data: observed data. Passed to Hypothesis.likelihood function
            proposals (list): A list of proposals, one for each chain
            temperatures (list): A list of floats, one for each chain. Temperature has to be positive.
            sampling_chain (list): Index of the chain to collect samples from. If not provided, chain with temperature
                1.0 is set as the sampling chain.
            burn_in (int): Number of burn-in iterations
            sample_count (int): Number of samples
            best_sample_count (int): Number of highest probability samples to keep
            thinning_period (int): Number of samples to discard before getting the next sample
            report_period (int): Number of iterations to report sampler status.
            verbose (bool):
        """
        if np.any(np.array(temperatures) <= 0):
            raise ValueError("Temperature has to be positive.")
        if len(proposals) != len(temperatures) or len(proposals) != len(initial_hs):
            raise ValueError("There should be one temperature, one initial hypothesis and one proposal "
                             "specified for each chain.")
        if sampling_chain is None and 1.0 not in temperatures:
            raise ValueError("sampling_chain has to be specified if no chain samples at temperature 1.0.")

        Sampler.__init__(self, initial_hs, data, proposals, burn_in, sample_count, best_sample_count, thinning_period,
                         report_period, verbose)

        self.chain_count = len(temperatures)
        self.temperatures = temperatures
        self.sampling_chain = sampling_chain
        if self.sampling_chain is None:
            self.sampling_chain = temperatures.index(1.0)

    def sample(self):
        """
        Sample from the posterior over hypotheses given data using PT algorithm
        """
        # assign the same initial hypothesis to all chain
        h = self.initial_h
        # proposed hypotheses
        hp = [None] * self.chain_count

        # log posterior of hypothesis
        log_p_h = [(h[i].log_prior() + h[i].log_likelihood(self.data)) / self.temperatures[i]
                   for i in range(self.chain_count)]
        # log posterior of proposed hypotheses
        log_p_hp = [0.0] * self.chain_count

        run = MCMCRun(info={"Sampler": "ParallelTemperingSampler", "Temperatures": self.temperatures},
                      log_row_count=(self.iter_count * (self.chain_count + 1)),
                      best_sample_count=self.best_sample_count,
                      log_columns=['Iteration', 'Chain', 'IsAccepted', 'LogProbability', 'LogAcceptanceRatio',
                                   'MoveType', 'Chain2', 'LogProbability2'])
        # Chain2 and LogProbability2 are used only for the exchange move

        accepted_count = [0 for i in range(self.chain_count)]
        exchange_accepted_count = 0
        print("ParallelTemperingSampler Start\n")
        for i in range(self.iter_count):

            # update each chain using MH
            for chain_id in range(self.chain_count):
                # propose next state
                move_type, hp[chain_id], q_hp_h, q_h_hp = self.proposal[chain_id].propose(h[chain_id])

                # calculate acceptance ratio
                # note that we already have log p(h), we only log p(hp)
                log_p_hp[chain_id] = (hp[chain_id].log_prior() + hp[chain_id].log_likelihood(self.data)) \
                                     / self.temperatures[chain_id]

                # a(h -> hp)
                log_a_hp_h = log_p_hp[chain_id] + np.log(q_h_hp) - (log_p_h[chain_id] + np.log(q_hp_h))

                is_accepted = 0
                # accept/reject
                if np.log(np.random.rand()) < log_a_hp_h:
                    if self.verbose:
                        print("Iteration {0:d}: T: {1:f} p {2:f} {3:f} a {4:f}. "
                              "Move: {5:s}\n".format(i, self.temperatures[chain_id],
                                                     log_p_h[chain_id], log_p_hp[chain_id], log_a_hp_h, move_type))

                    is_accepted = 1
                    accepted_count[chain_id] += 1
                    h[chain_id] = hp[chain_id]
                    log_p_h[chain_id] = log_p_hp[chain_id]

                run.record_log({"Iteration": i, "Chain": chain_id, "IsAccepted": is_accepted,
                                "LogProbability": log_p_h[chain_id], "LogAcceptanceRatio": log_a_hp_h,
                                "MoveType": move_type})
            # END update each chain

            # propose exchange move
            # we only propose exchanges between neighboring chains
            chain1 = np.random.randint(self.chain_count)
            if chain1 == 0:
                chain2 = 1
            elif chain1 == self.chain_count - 1:
                chain2 = chain1 - 1
            else:
                if np.random.rand() < .5:
                    chain2 = chain1 - 1
                else:
                    chain2 = chain1 + 1

            # exchange move
            # calculate exchange move acceptance ratio
            # a(c1,c2->c2,c1) = (p(h2)^(1/T1) * p(h1)^(1/T2) * q(c1<->c2)) / (p(h1)^(1/T1) * p(h2)^(1/T2) * q(c2<->c1))
            #                 = [p(h2) / p(h1))^(1/T1 - 1/T2)] * [q(c1<->c2) / q(c2<->c1)]
            # Note that q(c1<->c2) = q(c2<->c1) in our case.
            log_a_c2_c1 = ((1.0 / self.temperatures[chain1]) - (1.0 / self.temperatures[chain2])) * \
                          ((log_p_h[chain2] * self.temperatures[chain2]) -
                           (log_p_h[chain1] * self.temperatures[chain1]))

            is_exchange_accepted = 0
            # accept/reject
            if np.log(np.random.rand()) < log_a_c2_c1:
                if self.verbose:
                    print("Iteration {0:d}: E: {0:d}-{1:d} T: {2:f} {3:f} p: {4:f} {5:f} a {6:f}. "
                          "Move: exchange\n".format(i, chain1, chain2, self.temperatures[chain1],
                                                    self.temperatures[chain2], log_p_h[chain1], log_p_h[chain2],
                                                    log_a_c2_c1))

                is_exchange_accepted = 1
                exchange_accepted_count += 1
                temp = h[chain1]
                h[chain1] = h[chain2]
                h[chain2] = temp
                log_p_h[chain1] = (h[chain1].log_prior() + h[chain1].log_likelihood(self.data)) / self.temperatures[
                    chain1]
                log_p_h[chain2] = (h[chain2].log_prior() + h[chain2].log_likelihood(self.data)) / self.temperatures[
                    chain2]

            run.record_log({"Iteration": i, "Chain": chain1, "IsAccepted": is_exchange_accepted,
                            "LogProbability": log_p_h[chain1], "LogAcceptanceRatio": log_a_c2_c1,
                            "MoveType": "exchange", "Chain2": chain2, "LogProbability2": log_p_h[chain2]})
            # END exchange move

            if i >= self.burn_in:
                # was it the exchange move that gave us this new sample?
                if (self.sampling_chain == chain1 or self.sampling_chain == chain2) and is_exchange_accepted:
                    move_type = "exchange"

                if (i % self.thinning_period) == 0:
                    run.add_sample(h[self.sampling_chain], log_p_h[self.sampling_chain], i, move_type)

                run.add_best_sample(h[self.sampling_chain], log_p_h[self.sampling_chain], i, move_type)

            # report sampler state
            if (i % self.report_period) == 0:
                print("Iteration {0:d}, current hypothesis".format(i))
                print("Log posterior probability: {0:e}".format(log_p_h[chain_id]))
                print(h[chain_id])

        run.finish()
        acc_ratios = [float(accepted_count[i]) / self.iter_count for i in range(self.chain_count)]
        print("Sampling finished. Acceptance ratios for each chain: {0:s}\n".format(acc_ratios))
        print("Exchange move acceptance ratio: {0:f}\n".format(float(exchange_accepted_count) / self.iter_count))
        return run
