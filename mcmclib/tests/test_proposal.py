"""
mcmclib

Unit tests for Proposal classes.
There isn't much to test here. We just test if DeterministicMixtureProposal loops through all moves sequentially and
whether RandomMixtureProposal proposes each move at least once eventually.

Created on Dec 2, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import unittest

from mcmclib.proposal import *


# dummy moves
def move1(h, params):
    return h, 1.0, 1.0


def move2(h, params):
    return h, 1.0, 1.0


def move3(h, params):
    return h, 1.0, 1.0


class ProposalTest(unittest.TestCase):
    def test_deterministic_proposal(self):
        p = DeterministicMixtureProposal(moves={'1': move1, '2': move2, '3': move3}, params=None)
        m1, hp, p_hp_h, p_h_hp = p.propose(None)
        m2, hp, p_hp_h, p_h_hp = p.propose(None)
        m3, hp, p_hp_h, p_h_hp = p.propose(None)
        moves = [m1, m2, m3]
        self.assertIn('1', moves)
        self.assertIn('2', moves)
        self.assertIn('3', moves)

    def test_random_proposal(self):
        p = RandomMixtureProposal(moves={'1': move1, '2': move2, '3': move3}, params=None)
        # generate moves a large number of times
        # we should see each move at least one
        moves = []
        for i in range(100):
            m, hp, p_hp_h, p_h_hp = p.propose(None)
            moves.append(m)
        self.assertIn('1', moves)
        self.assertIn('2', moves)
        self.assertIn('3', moves)
