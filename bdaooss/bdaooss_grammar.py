"""
Big data analysis of object shape representations
Part-based shape grammar implementation

Created on May 27, 2015

@author: goker erdogan
"""

from scipy.misc import comb

from i3d.pcfg_tree import *
from i3d.shape_grammar import ShapeGrammarState, SpatialModel

"""
Definition of BDAoOSS Probabilistic Context Free Shape Grammar
The only part primitive is a rectangular prism and parts are docked to one 
of the six faces of its parent part. Each P is associated with one part
and `Null` is the terminal end symbol denoting an empty part.
"""
terminals = ['Null']
nonterminals = ['P']
start_symbol = 'P'
rules = {'P': [['Null'], ['P'], ['P', 'P'], ['P', 'P', 'P']]}
prod_probabilities = {'P': [.25, .25, .25, .25]}
terminating_rule_ids = {'P': [0]}
# maximum number of child nodes
MAX_CHILDREN = 3

bdaooss_shape_pcfg = PCFG(terminals, nonterminals, start_symbol, rules, prod_probabilities, terminating_rule_ids)

# viewpoint distance. canonical view is from xyz=(c, -c, c)
VIEWPOINT_RADIUS = np.sqrt(100.0)

# possible docking faces.
FACES = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
OPPOSITE_FACES = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}  # the opposite of each face
FACE_COUNT = 6
# used to denote that the part is not docked to any face. e.g., root node
NO_FACE = -1


# different size priors

def size_prior_child_smaller(parent_size):
    """
    This size prior constrains the child part to smaller in size than its parent.
    This is the original prior used when first generating objects.
    """
    # root part has a fixed size
    if parent_size is None:
        return np.array([1.5, 1.0, 1.0])

    # new part is always smaller than its parent in size
    w = np.random.uniform(max(0.1, parent_size[0] - .5), max(0.2, parent_size[0] - .2))
    d = np.random.uniform(max(0.1, parent_size[1] - .5), max(0.2, parent_size[1] - .2))
    h = np.random.uniform(max(0.1, parent_size[2] - .5), max(0.2, parent_size[2] - .2))
    size = np.array([w, d, h])
    return size


def size_prior_constant(parent_size):
    """
    All parts are the same size.
    """
    return np.array([1.0, 1.0, 1.0])


def size_prior_independent(parent_size):
    """
    The part sizes are randomly chosen from a uniform distribution independent
    of what the parent's size is.
    """
    # root part has a fixed size
    if parent_size is None:
        return np.array([1.5, 1.0, 1.0]) / np.sqrt(8.0)

    w = np.random.uniform(0.2, 1.5)
    d = np.random.uniform(0.2, 1.0)
    h = np.random.uniform(0.2, 1.0)
    size = np.array([w, d, h]) / np.sqrt(8.0)
    return size


def size_prior_infer3dshape(parent_size):
    """
    This prior is used in Infer3DShape. It is basically the
    same with the independent prior; just the range is [0.02, 1.02]
    """
    # root part has a fixed size
    if parent_size is None:
        return np.array([1.5, 1.0, 1.0]) / np.sqrt(8.0)
    return np.random.rand(3) + 0.02


class BDAoOSSSpatialState:
    """
    Spatial state class for BDAoOSS shape grammar.
    BDAoOSS spatial model uses this class to hold
    spatial information for nodes (parts) in the
    object.
    we hold the width (x), depth (y), and height (z) 
    of the part, to which face of its parent that 
    the part is docked to and the location of
    its center (xyz),"""

    def __init__(self, size=None, dock_face=None, position=None, occupied_faces=None):
        """
        Initialize spatial state for node
        If any of the parameters are not given, it
        initializes to default values of size 1, docking
        face NO_FACE, and position (0,0,0)
        """
        if size is None or dock_face is None or position is None or occupied_faces is None:
            self.size = np.array([1.0, 1.0, 1.0])
            self.dock_face = NO_FACE
            self.position = np.array([0.0, 0.0, 0.0])
            self.occupied_faces = []
        else:
            self.size = size
            self.dock_face = dock_face
            self.position = position
            self.occupied_faces = occupied_faces

    def copy(self):
        ss = BDAoOSSSpatialState(size=self.size.copy(), dock_face=self.dock_face,
                                 position=self.position.copy(), occupied_faces=self.occupied_faces[:])
        return ss

    def __eq__(self, comp):
        if np.sum(np.abs(self.position - comp.position)) < 1e-6 and np.sum(np.abs(self.size - comp.size)) < 1e-6:
            return True
        return False


class BDAoOSSSpatialModel(SpatialModel):
    """
    Spatial Model Class for BDAoOSS Shape Grammar
    For each P node holds the spatial state instance.
    """

    def __init__(self, size_prior=None, spatial_states=None):
        """
        Initialize spatial model

        """
        if spatial_states is None:
            self.spatial_states = {}
        else:
            self.spatial_states = spatial_states

            # prior for size
        # this is one of the functions defined at the top of this file.
        # self.size_prior = size_prior_independent # used for generating stimuli
        self.size_prior = size_prior
        if self.size_prior is None:
            self.size_prior = size_prior_infer3dshape

    def update(self, tree, grammar):
        """
        Updates spatial model, removes nodes that are not in
        nodes parameter and samples sizes, docking faces and
        positions for newly added nodes 
        """
        new_nodes = [node for node in tree.expand_tree(mode=Tree.WIDTH)
                     if tree[node].tag.symbol != 'Null']
        old_nodes = self.spatial_states.keys()
        removed_nodes = [node for node in old_nodes if node not in new_nodes]
        added_nodes = [node for node in new_nodes if node not in old_nodes]

        for n in removed_nodes:
            # NOTE occupied faces of parent is already updated in remove_part
            # OF COURSE, THIS IS QUITE MESSY. ALL THESE SHOULD BE DONE AT A SINGLE PLACE.
            del self.spatial_states[n]
        for n in added_nodes:
            if tree[n].bpointer is None:  # root node
                parent_sstate = None
            else:
                parent_sstate = self.spatial_states[tree[n].bpointer]

            self.spatial_states[n] = self._get_random_spatial_state(parent_sstate)

        # update positions
        for n in tree.expand_tree(mode=Tree.WIDTH):
            if tree[n].tag.symbol == 'P':
                nsstate = self.spatial_states[n]
                if tree[n].bpointer is not None:
                    npsstate = self.spatial_states[tree[n].bpointer]
                    nsstate.position = npsstate.position + \
                                       (FACES[nsstate.dock_face, :] * (nsstate.size + npsstate.size) / 2)

    def propose(self, tree, grammar):
        """ 
        Proposes a new spatial model based on current one.
        Creates a new spatial model with current node states,
        updates it, and returns it
        """
        sstates_copy = deepcopy(self.spatial_states)
        proposed_spatial_model = BDAoOSSSpatialModel(sstates_copy)
        proposed_spatial_model.update(tree, grammar)
        return proposed_spatial_model

    def _get_random_spatial_state(self, parent_sstate):
        """
        Returns a random spatial state based on the parent
        state's spatial state.
        Samples size, docking face randomly, and calculates
        position.
        """
        # empty constructor creates spatial state for root node.
        new_state = BDAoOSSSpatialState()
        # set the size of part
        parent_size = parent_sstate.size if parent_sstate is not None else None
        size = self.size_prior(parent_size)
        new_state.size = size

        if parent_sstate is not None:
            # find all empty faces of the parent part
            available_faces = [f for f in range(FACE_COUNT) if f not in parent_sstate.occupied_faces]

            """
            This is used when generating stimuli.
            # if the parent is root, don't dock to its bottom face
            if parent_sstate.dock_face == NO_FACE: # this is one way of checking if parent is the root.
                try:
                    available_faces.remove(5) # 5 is the bottom face
                except:
                    pass
            """

            face_ix = np.random.choice(available_faces)
            # face = FACES[face_ix]
            # update parent's occupied faces
            parent_sstate.occupied_faces.append(face_ix)
            # calculate position based on parent's position
            position = parent_sstate.position + (FACES[face_ix, :] * (size + parent_sstate.size) / 2)
            new_state.dock_face = face_ix
            new_state.position = position
            # update this part's occupied faces
            # the opposite of dock_face is now occupied
            new_state.occupied_faces = [OPPOSITE_FACES[face_ix]]

        return new_state

    def probability(self):
        """
        Returns probability of model
        """
        # we are assuming that each part picks one docking face from its parent's available faces. then we can go
        # through each spatial state (i.e., node), look at how many of its faces are occupied (i.e., how many children
        # it has) and calculate the prior probability. We need to take into account the possible orderings of the
        # children in the tree as well. Since reordering the children does not change the hypothesis, the prior
        # probability of picking the faces for a node's children is simply 1 / ( 6 choose number of children), 6 being
        # the number of docking faces of a node.
        # Note that we do not need to worry about the prior for size assignments because we have assumed them to be
        # uniform in [0, 1].
        p = 1.0
        for sstate in self.spatial_states.values():
            # we need to subtract 1 from the number of occupied faces because one
            # face is occupied by the parent (docking location). however, this is
            # not true for the root node because the root node does not have a
            # parent
            # similarly, there are 6 possible faces for a child of the root node, but for all other parent nodes,
            # there are 5 since 1 of the faces is occupied by the docking to the parent.
            available_face_count = FACE_COUNT
            child_count = len(sstate.occupied_faces)
            if sstate.dock_face != NO_FACE:  # if not root
                child_count -= 1
                available_face_count -= 1

            if child_count > 0:
                p *= (1.0 / comb(FACE_COUNT, child_count))
        return p

    def copy(self):
        # NOTE that this copy operation assumes that the node ids of a tree stay the
        # same when it is copied. This is the case for treelib trees.
        states = {}
        for n, ss in self.spatial_states.iteritems():
            states[n] = ss.copy()

        return BDAoOSSSpatialModel(size_prior=self.size_prior, spatial_states=states)

    def __str__(self):
        repr_str = "PartName  Size                Position            OccupiedFaces\n"
        fmt = "P         {0:20}{1:20}{2:20}\n"
        for key, state in self.spatial_states.iteritems():
            repr_str = repr_str + fmt.format(np.array_str(state.size, precision=2),
                                             np.array_str(state.position, precision=2),
                                             str(state.occupied_faces))
        return repr_str

    def __repr__(self):
        repr_str = "PartName  Size                Position            OccupiedFaces\n"
        fmt = "P         {0:20}{1:20}{2:20}\n"
        for key, state in self.spatial_states.iteritems():
            repr_str = repr_str + fmt.format(np.array_str(state.size, precision=2),
                                             np.array_str(state.position, precision=2),
                                             str(state.occupied_faces))
        return repr_str


class BDAoOSSShapeState(ShapeGrammarState):
    """BDAoOSS shape state class for BDAoOSS grammar and spatial model """

    def __init__(self, forward_model=None, data=None, ll_params=None, spatial_model=None, initial_tree=None,
                 viewpoint=None):
        """ Constructor for BDAoOSSShapeState Note that the first parameter ``grammar`` of 
        base class ShapeGrammarState is removed because this class is a grammar specific 
        implementation.  The additional parameter viewpoint determines the viewing point, 
        i.e., from which point in 3D space we look at the object. 
        Here we use spherical coordinates to specify it, i.e., (radius, polar angle, azimuth angle)."""
        self.MAXIMUM_DEPTH = 3

        if viewpoint is None:
            viewpoint = [VIEWPOINT_RADIUS, 0.0, 0.0]

        self.viewpoint = viewpoint

        ShapeGrammarState.__init__(self, grammar=bdaooss_shape_pcfg, forward_model=forward_model,
                                   data=data, ll_params=ll_params, spatial_model=spatial_model,
                                   initial_tree=initial_tree)

        # all the other functionality is independent of grammar and spatial model. 
        # hence they are implemented in ShapeGrammarState base class
        # grammar and spatial model specific methods are implemented below

    def convert_to_parts_positions(self):
        """
        Converts the state representation to parts and positions
        representation that can be given to forward model
        """
        parts = []
        positions = []
        scales = []
        for node, state in self.spatial_model.spatial_states.iteritems():
            parts.append(self.tree[node].tag.symbol)
            positions.append(state.position)
            scales.append(state.size)
        return parts, positions, scales, self.viewpoint

    def copy(self):
        """
        Returns a copy of the object.
        """
        tree = deepcopy(self.tree)
        sm_copy = self.spatial_model.copy()
        return BDAoOSSShapeState(forward_model=self.forward_model, data=self.data, ll_params=self.ll_params,
                                 spatial_model=sm_copy, initial_tree=tree, viewpoint=self.viewpoint)

    def probability(self):
        # everytime we add a new P part, probability goes down by 1/4.
        return (1.0 / 4.0) ** (len(self.spatial_model.spatial_states))

    def _likelihood(self):
        """
        Overwrite the likelihood in base PCFGTree class.
        We don't want that method to be called because we will
        implement our own likelihood when we need it.
        """
        pass

    def _get_nodes_at_depth(self, depth=1):
        """
        Returns a list of nodes at a given depth in the tree.
        """
        if depth < 0:
            raise ValueError('Depth cannot be negative')
        tree = self.tree
        depths = {}
        nodes = []
        for node in tree.expand_tree(mode=Tree.WIDTH):
            if tree[node].tag.symbol == 'P':
                # if root, depth is 0
                if tree[node].bpointer is None:
                    depths[node] = 0
                else:
                    depths[node] = depths[tree[node].bpointer] + 1

                cdepth = depths[node]
                if depths[node] == depth:
                    nodes.append(node)
                elif depths[node] > depth:
                    # since we are doing a breadth first expansion,
                    # we can quit when we get to a deeper level than
                    # desired.
                    break
        return nodes

    def add_part(self, parent_node):
        tree = self.tree
        sm = self.spatial_model
        # if the node has only Null as a child, we need to remove that Null
        # when we add a new part to it.
        if len(tree[parent_node].fpointer) == 1 and tree[tree[parent_node].fpointer[0]].tag.symbol == 'Null':
            tree.remove_node(tree[parent_node].fpointer[0])

        # add the new node to tree
        new_node = tree.create_node(tag=ParseNode('P', 0), parent=parent_node)
        # add a child Null node
        tree.create_node(tag=ParseNode('Null', ''), parent=new_node.identifier)

        # update parent's used production rule
        tree[parent_node].tag.rule += 1
        # update spatial model
        sm.update(tree, bdaooss_shape_pcfg)

    def remove_part(self, node_to_remove):
        tree = self.tree
        sm = self.spatial_model
        # remove node from tree (children nodes are also removed by the method)
        parent_node = tree[node_to_remove].bpointer
        tree.remove_node(node_to_remove)

        # if this node was the only child of its parent, we need to add a Null
        # node to the parent.
        # if this node was not the only child of its parent, we only need to
        # update the production rule used in the parent.
        if len(tree[parent_node].fpointer) == 0:
            new_node = tree.create_node(tag=ParseNode('Null', ''), parent=parent_node)
            tree[parent_node].tag.rule = 0
        else:
            tree[parent_node].tag.rule -= 1

        # update parent's occupied faces
        sm.spatial_states[parent_node].occupied_faces.remove(sm.spatial_states[node_to_remove].dock_face)

        # update spatial model
        sm.update(tree, bdaooss_shape_pcfg)

    def change_part_size(self, node, min_size=np.array([0.02, 0.02, 0.02]), max_size=np.array([1.02, 1.02, 1.02])):
        tree = self.tree
        sm = self.spatial_model

        # assign a new random size to part
        sm.spatial_states[node].size = np.random.uniform(min_size, max_size)
        # update part's and its children's positions
        sm.update(tree, bdaooss_shape_pcfg)

    def change_part_dock_face(self, node):
        tree = self.tree
        sm = self.spatial_model

        parent_node = tree[node].bpointer
        sstate = sm.spatial_states[node]
        parent_sstate = sm.spatial_states[parent_node]
        # get parent's occupied faces
        pofaces = parent_sstate.occupied_faces
        # available faces are the unoccupied faces of the parent but
        # we need to make sure that when we move the part, it does not
        # clash with one of the child parts of current node.
        # therefore, we remove the opposite faces of occupied faces
        # of our node from the list of available faces too.
        oofaces = sstate.occupied_faces
        oofaces = [OPPOSITE_FACES[f] for f in oofaces]
        # create a list of the available faces by removing occupied
        # faces
        afaces = [f for f in range(FACE_COUNT) if f not in pofaces and f not in oofaces]
        # if there are no available faces, don't move the part.
        if len(afaces) == 0:
            return

        # pick one randomly from the available faces
        face = np.random.choice(afaces)

        # update the occupied_faces of part, set the
        # old dock_face's opposite face to new dock face's
        # opposite
        sstate.occupied_faces[sstate.occupied_faces.index(OPPOSITE_FACES[sstate.dock_face])] = OPPOSITE_FACES[face]

        # update parent's occupied_faces, set the old
        # dock_face to new dock_face
        parent_sstate.occupied_faces[parent_sstate.occupied_faces.index(sstate.dock_face)] = face

        # update the dock_face of part
        sstate.dock_face = face

        # update part's and its children's positions
        sm.update(tree, bdaooss_shape_pcfg)

    """The following methods are used for generating stimuli for the experiment."""

    def _stimuli_add_part(self, depth=1):
        """
        Adds a new part to the tree at given depth.
        """
        if depth < 1:
            raise ValueError('Cannot add part to depth<1')
        # if we want to add a part to depth x, we need to find
        # a part at depth x-1
        # get nodes at given depth with available faces
        depth = depth - 1
        tree = self.tree
        sm = self.spatial_model
        nodes = self._get_nodes_at_depth(depth=depth)

        # if we want to add a part to a node, it should have less
        # than MAX_CHILDREN
        nodes = [node for node in nodes if len(tree[node].fpointer) < MAX_CHILDREN]

        try:
            node = np.random.choice(nodes)
        except ValueError:
            raise ValueError('No nodes to choose from.')

        self.add_part(parent_node=node)

    def _stimuli_remove_part(self, depth=1):
        """
        Removes a part from the tree at given depth.
        """
        if depth < 1:
            raise ValueError('Cannot remove part from depth<1')
        # get nodes at given depth 
        tree = self.tree
        sm = self.spatial_model
        nodes = self._get_nodes_at_depth(depth=depth)

        try:
            node = np.random.choice(nodes)
        except ValueError:
            raise ValueError('No nodes to choose from.')

        self.remove_part(node_to_remove=node)

    def _stimuli_move_part(self, depth=1):
        """
        Moves a part from the given depth to a different depth.
        Note that moving to the same depth will simply amount
        to changing the docking face of a part.
        Here we move the part to a different depth.
        Currently, depth 1 is moved to depth 2 and vice versa.
        """
        if depth < 1:
            raise ValueError('Cannot move part from depth<1')

        fromdepth = depth
        if fromdepth == 1:
            todepth = 2
        elif fromdepth == 2:
            todepth = 1
        else:
            raise ValueError('Depth should be either 1 or 2.')

        # get nodes at fromdepth
        tree = self.tree
        sm = self.spatial_model

        fromnodes = self._get_nodes_at_depth(depth=fromdepth)
        try:
            fromnode = np.random.choice(fromnodes)
        except ValueError:
            raise ValueError('No fromnodes to choose from.')

        parent_node = tree[fromnode].bpointer

        # get nodes at todepth
        tonodes = self._get_nodes_at_depth(depth=todepth)
        # if we want to add a part to a node, it should have less
        # than MAX_CHILDREN
        tonodes = [node for node in tonodes if len(tree[node].fpointer) < MAX_CHILDREN]
        # we also want to remove fromnode's parent from tonode, to make sure that
        # we indeed move the part (not simply change its docking face).
        # this can happen only when we are moving a part from depth x+1 to x.
        tonodes = [node for node in tonodes if node != parent_node]
        # we can't move a part to its child part. we need to remove
        # all descendants of fromnode from possible tonodes.
        tonodes = [node for node in tonodes if tree[node].bpointer != fromnode]

        try:
            tonode = np.random.choice(tonodes)
        except ValueError:
            raise ValueError('No tonodes to choose from.')

        # if the tonode has only Null as a child, we need to remove that Null
        # when we add a new part to it.
        if len(tree[tonode].fpointer) == 1 and tree[tree[tonode].fpointer[0]].tag.symbol == 'Null':
            tree.remove_node(tree[tonode].fpointer[0])

        # move fromnode to tonode
        tree.move_node(fromnode, tonode)

        tree[tonode].tag.rule = tree[tonode].tag.rule + 1
        tree[parent_node].tag.rule = tree[parent_node].tag.rule - 1

        # find available faces on tonode
        sstate = self.spatial_model.spatial_states[fromnode]
        parent_sstate = self.spatial_model.spatial_states[parent_node]
        to_sstate = self.spatial_model.spatial_states[tonode]

        # get parent's occupied faces
        pofaces = parent_sstate.occupied_faces
        # get tonode's occupied faces
        tofaces = to_sstate.occupied_faces

        # because we are moving the fromnode, its docking face is 
        # no longer occupied. 
        sstate.occupied_faces.remove(OPPOSITE_FACES[sstate.dock_face])
        # update parent's occupied_faces
        # remove fromnode's dock face from occupied faces
        parent_sstate.occupied_faces.remove(sstate.dock_face)

        # available faces are the unoccupied faces of the tonode but
        # we need to make sure that when we move the part, it does not
        # clash with one of the child parts of current node.
        # therefore, we remove the opposite faces of occupied faces
        # of our node from the list of available faces too. 
        oofaces = sstate.occupied_faces
        oofaces = [OPPOSITE_FACES[f] for f in oofaces]
        # create a list of the available faces by removing occupied
        # faces
        afaces = [f for f in range(FACE_COUNT) if f not in tofaces and f not in oofaces]
        # pick one randomly from the available faces
        face = np.random.choice(afaces)

        # update the occupied_faces of part
        # set the new dock face
        # add the new dock_face to occupied.
        sstate.occupied_faces.append(OPPOSITE_FACES[face])
        sstate.dock_face = face

        # update tonode's occupied faces. add the new dock face to occupied.
        to_sstate.occupied_faces.append(face)

        # if this fromnode was the only child of its parent, we need to add a Null
        # node to the parent.
        # if this fromnode was not the only child of its parent, we only need to
        # update the production rule used in the parent.
        if len(tree[parent_node].fpointer) < 1:
            new_node = tree.create_node(tag=ParseNode('Null', ''), parent=parent_node)

        # update part's and its children's positions
        for n in tree.expand_tree(mode=Tree.WIDTH):
            if tree[n].tag.symbol == 'P' and tree[n].bpointer is not None:
                nsstate = self.spatial_model.spatial_states[n]
                npsstate = self.spatial_model.spatial_states[tree[n].bpointer]
                nsstate.position = npsstate.position + (
                            FACES[nsstate.dock_face, :] * (nsstate.size + npsstate.size) / 2)

    def _stimuli_vary_part_size(self, depth=1):
        """Pick a part randomly at given depth and change its size.
        """
        tree = self.tree
        nodes = self._get_nodes_at_depth(depth=depth)
        try:
            node = np.random.choice(nodes)
        except ValueError:
            raise ValueError('No nodes to choose from.')

        # max size is parent's size
        pnode = tree[node].bpointer
        max_size = np.array([1.5, 1, 1])
        if pnode is not None:
            max_size = self.spatial_model.spatial_states[pnode].size
        # min size is max of child's size
        cnodes = [n for n in tree[node].fpointer if tree[n].tag.symbol != 'Null']
        min_size = np.array([.2, .2, .2])
        for cnode in cnodes:
            min_size = np.max(np.vstack([min_size, self.spatial_model.spatial_states[cnode].size]), 0)

        self.change_part_dock_face(node, min_size=min_size, max_size=max_size)

    def _stimuli_vary_dock_face(self, depth=1):
        """Pick a part randomly at a given depth and 
        change the face it is docked to.
        Performs the change in place.
        """
        tree = self.tree
        if depth < 1:  # can't change the dock_face of root
            raise ValueError('Depth must be greater than 0')

        # get nodes at given depth
        nodes = self._get_nodes_at_depth(depth=depth)

        try:
            node = np.random.choice(nodes)
        except ValueError:
            raise ValueError('No nodes to choose from. Provided depth > tree depth?')

        self.change_part_dock_face(node)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        raise NotImplementedError()

    def __neq__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        repr_str = "Id                  Size                Position            DockFace  OccupiedFaces\n"
        fmt = "{0:20.15}{1:20}{2:20}{3:10}{4:20}\n"
        for node in self.tree.expand_tree(mode=Tree.WIDTH):
            if self.tree[node].tag.symbol != 'Null':
                repr_str = repr_str + fmt.format(node, np.array_str(self.spatial_model.spatial_states[node].size,
                                                                    precision=2),
                                                 np.array_str(self.spatial_model.spatial_states[node].position,
                                                              precision=2),
                                                 str(self.spatial_model.spatial_states[node].dock_face),
                                                 str(self.spatial_model.spatial_states[node].occupied_faces))

        repr_str = repr_str + "\n\nTree Structure\nNode                Children\n"
        fmt = "{0:20.15}{1:s}\n"
        for node in self.tree.expand_tree(mode=Tree.WIDTH):
            if self.tree[node].tag.symbol != 'Null':
                repr_str = repr_str + fmt.format(node, "".join(
                    ['{0:15.10}'.format(child) for child in self.tree[node].fpointer]))

        return repr_str

    def __str__(self):
        return repr(self)

    def __getstate__(self):
        """
        Return data to be pickled. 
        We only need the tree, spatial model, viewpoint, data and ll_params
        """
        state = {}
        state['tree'] = self.tree
        state['spatial_model'] = self.spatial_model
        state['data'] = self.data
        state['ll_params'] = self.ll_params
        state['viewpoint'] = self.viewpoint
        state['grammar'] = self.grammar
        return state
