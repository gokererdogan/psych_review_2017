# -*- coding: utf-8 -*-
"""
Created on May 14, 2013
Analysis of Multisensory Representations
Base classes for shape grammar state and 
shape spatial model.
@author: gerdogan
Last Update: May 14, 2013
"""

import numpy as np

from pcfg_tree import PCFGTree
from treelib import Tree


class SpatialModel:
    """
    Spatial Model Base Class for Shape Grammar
    Subclass this class to implement your own spatial
    model
    """

    def __init__(self):
        """
        Initialize spatial model
        If positions is not given, it is simply instantiated
        with an empty dictionary
        """
        pass

    def update(self, data):
        """
        Updates spatial model 
        """
        pass

    def propose(self, data):
        """
        Proposes a new spatial model based on current one.
        """
        pass

    def probability(self):
        """
        Returns probability of model
        """
        pass


class ShapeGrammarState(PCFGTree):
    """
    State representation class for shape grammar
    Contains parse tree of representation in self.tree
    and spatial layout in self.spatial_model
    This class implements general functionality for any shape
    grammar, you should implement new classes that inherit this
    base class if you need more functionality.
    """
    # maximum depth allowed for a parse tree
    # get_random_tree returns trees that have depth =< MAX_DEPTH
    # reason for implementing this constraint is the fact that for
    # some grammars (where branching factor is high) the tree grows
    # without bound
    MAXIMUM_DEPTH = 3

    def __init__(self, grammar, forward_model=None, data=None, ll_params=None, spatial_model=None, initial_tree=None):
        """
        Initializes ShapeGrammarState
        grammar: Probabilistic context free shape grammar definition. PCFG instance.
        forward_model: Forward model(s) used in likelihood calculation
        ll_params: parameters for likelihood calculation
        spatial_model: Spatial model for the state. Initialized randomly if initial_tree is not
                        provided
        initial_tree (optional): Parse tree for the state. If not provided, state is initialized
                        randomly.
        """
        self.grammar = grammar
        self.spatial_model = spatial_model
        self.forward_model = forward_model

        if initial_tree is None:
            self.tree = self._get_random_tree(start=self.grammar.start_symbol, max_depth=self.MAXIMUM_DEPTH)
            # initialize spatial model
            self.spatial_model.update(self.tree, self.grammar)
        else:
            self.tree = initial_tree

        # if a subclass does not define moves, we define it here and add subtree move
        if hasattr(self, 'moves') is False:
            self.moves = [self.subtree_proposal]

        # IMPORTANT: we call base class init after we initialize spatial model, 
        # because prior, ll, deriv_prob calculation is done in init and
        # spatial_model should be available to calculate them
        PCFGTree.__init__(self, grammar=grammar, data=data, ll_params=ll_params, initial_tree=self.tree)

    def subtree_proposal(self):
        """
        Proposes a new state based on current state using subtree move
        Uses base class PCFGTree's subtree_proposal_propose_tree to generate 
        new tree, then samples positions for new added parts
        """
        pcfg_proposal = PCFGTree.subtree_proposal_propose_tree(self)
        # get a new spatial model based on proposed tree
        proposed_spatial_model = self.spatial_model.propose(pcfg_proposal, self.grammar)
        proposal = self.__class__(forward_model=self.forward_model, data=self.data, ll_params=self.ll_params,
                                  spatial_model=proposed_spatial_model, initial_tree=pcfg_proposal)
        acc_prob = self._subtree_acceptance_probability(proposal)
        return proposal, acc_prob

    def _subtree_acceptance_probability(self, proposal):
        """
        Acceptance probability for subtree move
        Note that acceptance probability is simply RationalRules acceptance
        probability (i.e. independent of spatial model)
        The derivation for this can be seen in the paper
        """
        # calculate acceptance probability
        acc_prob = 1
        nt_current = [node for node in self.tree.expand_tree(mode=Tree.WIDTH)
                      if self.tree[node].tag.symbol in self.grammar.nonterminals]
        nt_proposal = [node for node in proposal.tree.expand_tree(mode=Tree.WIDTH)
                       if proposal.tree[node].tag.symbol in self.grammar.nonterminals]

        # prior terms contain prior probabilities for spatial model too, so
        # in order to get back to Rational Rules prior we multiply with
        # inverse of spatial model probabilities
        acc_prob = acc_prob * proposal.prior * proposal.likelihood * len(
            nt_current) * self.derivation_prob * self.spatial_model.probability()
        acc_prob = acc_prob / (self.prior * self.likelihood * len(
            nt_proposal) * proposal.derivation_prob * proposal.spatial_model.probability())
        return acc_prob

    def _prior(self):
        """
        Prior probability for state
        Product of probability for tree and spatial layout
        """
        prior = PCFGTree._prior(self) * self.spatial_model.probability()

        return prior

    def _likelihood(self):
        """
        Likelihood function
        Gets render from forward model and calculates distance between
        data and render.
        """
        data = self.data
        params = self.ll_params
        b = params['b']
        render = self.forward_model.render(self)
        distance = np.sum((render - data) ** 2) / float(render.size)
        return np.exp(-b * distance)

    def convert_to_parts_positions(self):
        """
        Converts the state representation to parts and positions
        representation that can be given to forward model
        Override in super class
        """
        pass

    def __eq__(self, other):
        """
        Override in super class if checking for equality is needed
        Equality checking is a grammar, and spatial model specific 
        operation.
        """
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        """
        Override in super class for more informative string representations
        """
        return PCFGTree.__repr__(self)

    def __str__(self):
        """
        Override in super class for more informative string representations
        """
        return PCFGTree.__str__(self)

    def __getstate__(self):
        """
        Return data to be pickled. 
        ForwardModel cannot be pickled because it contains VTKObjects, similarly
        moves cannot be pickled because it contains instancemethod objects. that's
        why we remove them from data to be pickled
        """
        return dict((k, v) for k, v in self.__dict__.iteritems() if k not in ['forward_model', 'moves'])
