"""
Created on May 14, 2013
Abstract class for Probabilistic Context Free Grammar Parse Tree
@author: gerdogan
"""
from copy import deepcopy

import numpy as np
import scipy.special as sp

from treelib import Tree


class PCFG:
    """
    Definition of Probabilistic Context Free Grammar
    Look into rational_rules.py for a sample grammar specification
    """

    def __init__(self, terminals, nonterminals, start_symbol, rules, prod_probabilities, terminating_rule_ids):
        self.terminals = terminals
        self.nonterminals = nonterminals
        self.start_symbol = start_symbol
        self.rules = rules
        self.prod_probabilities = prod_probabilities
        self.terminating_rule_ids = terminating_rule_ids


class ParseNode:
    """
    Represents a node in parse tree
    A simple class that contains the symbol from the
    grammar and the index of the production rule used to generate
    the children of the node 
    """

    def __init__(self, symbol='', rule=''):
        self.symbol = symbol
        self.rule = rule

    def __str__(self):
        return self.symbol + ' ' + repr(self.rule)

    def __deepcopy(self):
        return ParseNode(symbol=self.symbol, rule=self.rule)


class PCFGTree:
    """
    PCFG Parse Tree for MCMC
    Prior and acceptance probabilities defined according to
    Reference: Goodman, N. D., Tenenbaum, J. B., Feldman, J., & Griffiths, T. L. (2008).
    A rational analysis of rule-based concept learning. Cognitive science, 32(1), 108-54.
    """

    def __init__(self, grammar, data=None, ll_params=None, initial_tree=None, maximum_depth=None):
        """
        grammar: PCFG object that defines the grammar
        data: Data, used for calculatng likelihood
        ll_params: Likelihood specific parameters
        """
        self.grammar = grammar
        self.data = data
        self.ll_params = ll_params
        # subclasses may define a maximum allowed tree depth
        # if there are no such definitions, we set a limit here
        if hasattr(self, 'MAXIMUM_DEPTH') is False:
            self.MAXIMUM_DEPTH = maximum_depth
            if self.MAXIMUM_DEPTH is None:
                self.MAXIMUM_DEPTH = 99
        if initial_tree is None:
            self.tree = self._get_random_tree(start=self.grammar.start_symbol, max_depth=self.MAXIMUM_DEPTH)
        else:
            self.tree = initial_tree

        # available moves (proposals)
        # subclasses may define their own moves, so we need to check if moves is alredy defined
        if hasattr(self, 'moves') is False:
            self.moves = [self.subtree_proposal]

        # WARNING
        # Note that we are calling self._prior() etc. here
        # that means we are calling super class's methods 
        # this means that we need to initialize any information
        # that are used in these methods before we call base 
        # class's init 
        self.prior = self._prior()
        self.derivation_prob = self._derivation_prob()
        self.likelihood = self._likelihood()

    def _get_random_tree(self, start, max_depth=999):
        """
        Returns a random tree from PCFG starting with symbol 'start'
        depth: the maximum depth of tree
        """
        t = Tree()
        t.create_node(ParseNode(start, ''))

        # get ids of not expanded nonterminals in tree
        nodes_to_expand, depth = self.__get_nodes_to_expand_and_depth(t)

        while len(nodes_to_expand) > 0:
            # for each non terminal, choose a random rule and apply it
            for node in nodes_to_expand:
                symbol = t[node].tag.symbol

                # if tree exceeded the allowed depth, expand nonterminals
                # using rules from terminating_rule_ids
                if depth >= (max_depth - 1):
                    # choose from rules for nonterminal from terminating_rule_ids
                    rhsix = np.random.choice(self.grammar.terminating_rule_ids[symbol], size=1)
                else:
                    # choose from rules for nonterminal according to production probabilities
                    rhsix = np.random.choice(len(self.grammar.rules[symbol]), p=self.grammar.prod_probabilities[symbol],
                                             size=1)

                t[node].tag.rule = rhsix[0]  # index of production rule used when expanding this node
                rhs = self.grammar.rules[symbol][rhsix[0]]
                for s in rhs:
                    t.create_node(tag=ParseNode(s, ''), parent=node)

            nodes_to_expand, depth = self.__get_nodes_to_expand_and_depth(t)

        return t

    def __get_nodes_to_expand_and_depth(self, tree):
        """
        Gets the nodes that should be expanded and the depth of
        tree. Used by _get_random_tree
        """
        nodes_to_expand = []
        depths = {}

        for node in tree.expand_tree(mode=Tree.WIDTH):
            # if node is a nonterminal and it has no children, it should be expanded
            if tree[node].tag.symbol in self.grammar.nonterminals and len(tree[node].fpointer) == 0:
                nodes_to_expand.append(node)

            # if root, depth is 0
            if tree[node].bpointer is None:
                depths[node] = 0
            else:
                depths[node] = depths[tree[node].bpointer] + 1

        return nodes_to_expand, max(depths.values())

    def get_depth(self):
        """
        Gets the depth of tree
        """
        depths = {}

        for node in self.tree.expand_tree(mode=Tree.WIDTH):
            # if root, depth is 0
            if self.tree[node].bpointer is None:
                depths[node] = 0
            else:
                depths[node] = depths[self.tree[node].bpointer] + 1

        return max(depths.values())

    def subtree_proposal_propose_tree(self):
        """
        Proposes a new tree based on current state's tree
        Chooses a non-terminal node randomly, prunes its subtree and
        replaces it with a random new subtree
        """
        proposed_state = deepcopy(self.tree)
        nonterminal_nodes = [node for node in proposed_state.expand_tree(mode=Tree.WIDTH)
                             if proposed_state[node].tag.symbol in self.grammar.nonterminals]
        chosen_node = np.random.choice(nonterminal_nodes)
        chosen_symbol = proposed_state[chosen_node].tag.symbol

        # get depth of current tree to find out the maximum allowed depth for
        # proposed tree
        ne, depth = self.__get_nodes_to_expand_and_depth(self.tree)
        max_depth = self.MAXIMUM_DEPTH - depth if (self.MAXIMUM_DEPTH - depth) > 0 else 1
        new_subtree = self._get_random_tree(chosen_symbol, max_depth)
        if chosen_node == proposed_state.root:
            proposed_state = new_subtree
        else:
            # Tree.paste method does not care about the order of children 
            # and appends the new subtree as the last child to parent_node
            # we want to paste the subtree to the exact location we pruned
            # that's why we do not use Tree.paste method
            parent_node_id = proposed_state[chosen_node].bpointer
            parent_node = proposed_state[parent_node_id]
            parent_child_location = parent_node.fpointer.index(chosen_node)
            proposed_state.remove_node(chosen_node)
            new_subtree[new_subtree.root].bpointer = parent_node_id
            parent_node.fpointer.insert(parent_child_location, new_subtree.root)
            proposed_state.nodes.update(new_subtree.nodes)

        return proposed_state

    def subtree_proposal(self):
        """
        Propose new state based on current state using subtree move
        Proposes a new tree using propose_tree function and
        instantiates a new instance of PCFGTree with it, 
        then returns it and its acceptance probability
        You should override this method if your state representation
        contains extra data other than tree 
        """
        # propose new state
        new_tree = self.subtree_proposal_propose_tree()
        proposal = self.__class__(self.grammar, data=self.data, ll_params=self.ll_params, initial_tree=new_tree)
        acc_prob = self._subtree_proposal_acceptance_probability(proposal)
        return proposal, acc_prob

    def _subtree_proposal_acceptance_probability(self, proposal):
        # calculate acceptance probability
        acc_prob = 1
        nt_current = [node for node in self.tree.expand_tree(mode=Tree.WIDTH)
                      if self.tree[node].tag.symbol in self.grammar.nonterminals]
        nt_proposal = [node for node in proposal.tree.expand_tree(mode=Tree.WIDTH)
                       if proposal.tree[node].tag.symbol in self.grammar.nonterminals]

        acc_prob = acc_prob * proposal.prior * proposal.likelihood * len(nt_current) * self.derivation_prob
        acc_prob = acc_prob / (self.prior * self.likelihood * len(nt_proposal) * proposal.derivation_prob)
        return acc_prob

    def _prior(self):
        """
        Prior probability for state
        Calculated using Eq. 13 in ref.
        Here we marginalize out the production rule probabilities assuming 
        they are all uniform
        """
        prior = 1.00
        nonterminal_nr = [[self.tree[node].tag.symbol, self.tree[node].tag.rule] for node in
                          self.tree.expand_tree(mode=Tree.WIDTH)
                          if self.tree[node].tag.symbol in self.grammar.nonterminals]
        used_nonterminals = set([nr[0] for nr in nonterminal_nr])
        for nt in used_nonterminals:
            used_rules = [nr[1] for nr in nonterminal_nr if nr[0] == nt]
            rule_counts = np.bincount(used_rules, minlength=len(self.grammar.rules[nt]))
            prior = prior * self.__mult_beta(rule_counts + np.ones_like(rule_counts))
            prior = prior / self.__mult_beta(np.ones_like(rule_counts))

        return prior

    def _derivation_prob(self):
        """
        Probability of a state given grammar and production probabilities
        Calculated using Eq. 3 in ref.
        Here we do not marginalize out production rule probabilities.
        This function is used for calculating acceptance probability.
        """
        prob = 1.00
        nonterminal_nr = [[self.tree[node].tag.symbol, self.tree[node].tag.rule] for node in
                          self.tree.expand_tree(mode=Tree.WIDTH)
                          if self.tree[node].tag.symbol in self.grammar.nonterminals]

        for nt, rule in nonterminal_nr:
            prob = prob * self.grammar.prod_probabilities[nt][rule]

        return prob

    def _likelihood(self):
        """
        Likelihood function
        Should be overridden in super class
        """
        pass

    def __mult_beta(self, vect):
        """
        Multinomial beta function (normalization term for Dirichlet)
        mbeta(x,y,z) = gamma(x)*gamma(y)*gamma(z) / gamma(x+y+z)
        """
        ret = 1.0
        for i in vect:
            ret = ret * sp.gamma(i)
        ret = ret / sp.gamma(np.sum(vect))
        return ret

    def __eq__(self, other):
        """
        Checks for equality between two trees
        Override this in superclass if you want to
        be able to check for equality between MCMC
        states. This is useful for MCMSSampler class 
        to find samples with highest probability. 
        """
        pass

    def __ne__(self, other):
        """
        Checks for inequality between two trees
        Override this in superclass if you want to
        be able to check for equality between MCMC
        states. This is useful for MCMSSampler class 
        to find samples with highest probability. 
        """
        pass
        pass

    def __repr__(self):
        return "".join(self.tree[node].tag.symbol for node in self.tree.expand_tree(mode=Tree.DEPTH) if
                       len(self.tree[node].fpointer) == 0)

    def __str__(self):
        return "".join(self.tree[node].tag.symbol for node in self.tree.expand_tree(mode=Tree.DEPTH) if
                       len(self.tree[node].fpointer) == 0)

    def __getstate__(self):
        """
        Return data to be pickled. 
        moves cannot be pickled because it contains instancemethod objects, that's
        why we remove it from data to be pickled
        """
        return dict((k, v) for k, v in self.__dict__.iteritems() if k is not 'moves')
