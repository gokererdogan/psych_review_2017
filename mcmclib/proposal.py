"""
mcmclib

This file contains the abstract Proposal class and two simple
Proposal classes.

Created on Aug 27, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import numpy as np


class Proposal(object):
    """
    Proposal class implements MCMC moves (i.e., proposals) on Hypothesis.
    This is an abstract class specifying the template for Proposal classes.
    Propose method is called by MCMCSampler to get the next proposed hypothesis.
    """

    def __init__(self, params):
        self.params = params

    def propose(self, h):
        """
        Proposes a new hypothesis based on h
        Returns (information string, new hypothesis, probability of move, probability of reverse move)
        """
        return NotImplementedError()


class RandomMixtureProposal(Proposal):
    """
    RandomMixtureProposal class implements a mixture kernel where one move is picked randomly
    from a set of given moves.
    Note that this is always a valid proposal strategy as long as the individual moves satisfy
    detailed balance.
    """

    def __init__(self, moves, params):
        """
        Initializes the mixture kernel with moves.
        :param moves: a dictionary of functions. each function implements a move proposing a new hypothesis based on the
        current one.
        :return: RandomMixtureProposal instance
        """
        Proposal.__init__(self, params)
        self.moves = moves

    def propose(self, h):
        """
        Pick one move randomly and use it to propose a new hypothesis based on current hypothesis h.
        :param h:  Current hypothesis
        :return: 4-tuple consisting of (move name, new hypothesis, p(new h|h), p(h|new h)
        """
        move = np.random.choice(self.moves.keys())
        hp, p_hp_h, p_h_hp = self.moves[move](h, self.params)
        return move, hp, p_hp_h, p_h_hp


class DeterministicMixtureProposal(Proposal):
    """
    DeterministicMixtureProposal class implements a mixture kernel where each move is picked sequentially from a set of
    given moves.
    Note that this is always a valid proposal strategy as long as the individual moves satisfy detailed balance.
    """

    def __init__(self, moves, params):
        """
        Initializes the mixture kernel with moves.
        :param moves: a dictionary of functions. each function implements a move proposing a new hypothesis based on the
        current one.
        :return: DeterministicMixtureProposal instance
        """
        Proposal.__init__(self, params)
        self.moves = moves
        self.move_names = self.moves.keys()
        self.move_count = len(moves)
        self.current_move = 0

    def propose(self, h):
        """
        Pick the next move and use it to propose a new hypothesis based on current hypothesis h.
        :param h:  Current hypothesis
        :return: 4-tuple consisting of (move name, new hypothesis, p(new h|h), p(h|new h)
        """
        move = self.move_names[self.current_move]
        hp, p_hp_h, p_h_hp = self.moves[move](h, self.params)
        self.current_move = (self.current_move + 1) % self.move_count
        return move, hp, p_hp_h, p_h_hp
