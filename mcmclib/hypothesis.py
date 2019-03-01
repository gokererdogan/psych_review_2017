"""
mcmclib

This file contains the abstract Hypothesis class.

Created on Aug 27, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""


class Hypothesis(object):
    """
    Hypothesis class is an abstract class that specifies the template
    for an MCMC hypothesis.
    """

    def __init__(self):
        """
        Hypothesis class constructor

        :params: Additional parameters
        """
        # log_p: log prior, log ll: log likelihood
        # we want to cache these values, therefore we initialize them to None
        # prior and ll methods should calculate these once and return p and ll
        self._log_p = None
        self._log_ll = None

    def _calculate_log_prior(self):
        """
        This method calculates the log prior probability of hypothesis.
        This method needs to be overridden in children classes.
        """
        return NotImplementedError()

    def _calculate_log_likelihood(self, data=None):
        """
        This method calculates the log likelihood of hypothesis.
        This method needs to be overridden in children classes.
        """
        return NotImplementedError()

    def log_prior(self):
        """
        Returns log prior probability log p(H) of the hypothesis
        """
        if self._log_p is None:
            self._log_p = self._calculate_log_prior()
        return self._log_p

    def log_likelihood(self, data=None):
        """
        Returns the log likelihood of hypothesis given data, p(D|H)
        """
        if self._log_ll is None:
            self._log_ll = self._calculate_log_likelihood(data)
        return self._log_ll

    def copy(self):
        """
        Returns a (deep) copy of the hypothesis. Used for generating
        new hypotheses based on itself.
        """
        return NotImplementedError()
