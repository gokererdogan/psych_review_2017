"""
Inferring 3D Shape from 2D Images

This file contains the BDAoOSSShapeMaxD class that is BDAoOSSShape with a maximum depth
constraint.

Created on Oct 6, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""
from copy import deepcopy

import numpy as np

from bdaooss_shape import BDAoOSSShape


class BDAoOSSShapeMaxD(BDAoOSSShape):
    """Shape hypothesis class based on BDAoOSS Shape grammar. This class assumes a maximum depth to parse trees and
    uses a new prior distribution over hypotheses.

    Note that this class in fact does not enforce the maximum depth constraint. That is achieved by the proposal
    functions. However, this new class enables us to define a new prior over hypotheses; that is the main reason we
    want to constrain tree depth. HOWEVER, currently this class uses the same prior with BDAoOSSShape; therefore, this
    class is almost unnecessary. However, because the proposal functions check the maximum depth by looking at the
    `max_depth` attribute, we need this class for now. We should probably get rid of this class by moving `max_depth`
     to BDAoOSSShape in the future.
    """

    def __init__(self, forward_model, shape=None, viewpoint=None, params=None, max_depth=10):
        """Initialize BDAoOSSShapeMaxD instance

        Parameters:
            max_depth (int): Maximum depth allowed for trees

        Returns:
            BDAoOSSShapeMaxD
        """
        BDAoOSSShape.__init__(self, forward_model=forward_model, shape=shape, viewpoint=viewpoint, params=params)

        # this is actually not needed anywhere.
        self.max_depth = max_depth

    def _calculate_log_prior(self):
        """Prior for the BDAoOSShapeMaxD hypothesis.

        Returns:
            (float): log prior for the hypothesis.
        """
        """
        # We initially wanted to use a uniform prior over number parts, here is how we can define it.
        # To assume a uniform prior over number of parts, we can let the derivation prob. for all trees to be equal.
        # Note that we still have to keep the spatial model probability as there are many hypotheses
        # with the same number of parts but with different spatial models.
        # NOTE we are not using this prior, because it will lead to chains that are potentially quite slow since
        # the chain will spend an equal amount of time sampling quite large trees as it does sampling small trees.
        return np.log(self.shape.spatial_model.probability())
        """
        # use the same prior with BDAoOSSShape
        return np.log(self.shape.probability()) + np.log(self.shape.spatial_model.probability())

    def copy(self):
        # NOTE that we are not copying params. This assumes that params do not change from
        # hypothesis to hypothesis.
        shape_copy = self.shape.copy()
        viewpoint_copy = deepcopy(self.viewpoint)
        return BDAoOSSShapeMaxD(forward_model=self.forward_model, shape=shape_copy, params=self.params,
                                max_depth=self.max_depth, viewpoint=viewpoint_copy)


if __name__ == "__main__":
    import mcmclib.proposal
    from i3d import i3d_proposal, vision_forward_model as vfm
    import bdaooss_shape as bd

    max_depth = 3

    fwm = vfm.VisionForwardModel(render_size=(200, 200), offscreen_rendering=False, custom_lighting=True)
    h = BDAoOSSShapeMaxD(forward_model=fwm, viewpoint=[(np.sqrt(2.0), -np.sqrt(2.0), 2.0)],
                         params={'LL_VARIANCE': 0.0001}, max_depth=max_depth)

    """
    moves = {'bdaooss_add_remove_part': bdaooss_add_remove_part, 'bdaooss_change_part_size': bdaooss_change_part_size,
             'bdaooss_change_part_size_local': bdaooss_change_part_size_local,
             'bdaooss_change_part_dock_face': bdaooss_change_part_dock_face,
             'bdaooss_move_object': bdaooss_move_object, 'change_viewpoint': i3d_proposal.change_viewpoint}
             """

    moves = {'bdaooss_add_remove_part': bd.bdaooss_add_remove_part,
             'bdaooss_change_part_size_local': bd.bdaooss_change_part_size_local,
             'bdaooss_change_part_dock_face': bd.bdaooss_change_part_dock_face,
             'change_viewpoint': i3d_proposal.change_viewpoint}

    params = {'MOVE_OBJECT_VARIANCE': 0.00005,
              'CHANGE_SIZE_VARIANCE': 0.00005,
              'CHANGE_VIEWPOINT_VARIANCE': 30.0,
              'MAX_DEPTH': max_depth}

    proposal = mcmclib.proposal.RandomMixtureProposal(moves, params)

    # data = np.load('data/test1_single_view.npy')
    data = np.load('data/stimuli20150624_144833/o1_single_view.npy')

    # choose sampler
    thinning_period = 2000
    sampler_class = 'mh'
    if sampler_class == 'mh':
        import mcmclib.mh_sampler

        sampler = mcmclib.mh_sampler.MHSampler(h, data, proposal, burn_in=1000, sample_count=10, best_sample_count=10,
                                               thinning_period=thinning_period, report_period=thinning_period)
    elif sampler_class == 'pt':
        from mcmclib.parallel_tempering_sampler import ParallelTemperingSampler

        sampler = ParallelTemperingSampler(initial_hs=[h, h, h], data=data, proposals=[proposal, proposal, proposal],
                                           temperatures=[3.0, 1.5, 1.0], burn_in=1000, sample_count=10,
                                           best_sample_count=10, thinning_period=int(thinning_period / 3.0),
                                           report_period=int(thinning_period / 3.0))
    else:
        raise ValueError('Unknown sampler class')

    run = sampler.sample()
