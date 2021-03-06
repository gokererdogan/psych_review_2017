"""
Inferring 3D Shape from 2D Images

This file contains the ShapeMaxN class which implements a
Shape object with maximum N parts. This is done in order to
assign a flat prior to all hypotheses. We don't want the
prior to favor objects with fewer parts.

Created on Sep 14, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

from shape import *


class ShapeMaxN(Shape):
    """
    ShapeMaxN class defines a 3D object with maximum N parts.

    This class is based on Shape class and defines only one additional attribute.

    Attributes:
        maxn (int): number of maximum parts allowed.
            Note this attribute is only used to make sure the randomly generated hypothesis does not have more than
            maxn parts. The mechanics of ensuring that a hypothesis never has more than maxn parts are handled by the
            proposal functions.
    """
    def __init__(self, forward_model, viewpoint=None, parts=None, params=None, maxn=10):
        Shape.__init__(self, forward_model=forward_model, viewpoint=viewpoint, params=params, parts=parts)
        self.maxn = maxn

        # generative process: add a new part until rand()>theta (add part prob.)
        # p(H|theta) = theta^|H| (1 - theta)
        if parts is None:
            # randomly generate a new shape
            parts = [CuboidPrimitive()]
            while (np.random.rand() < ADD_PART_PROB) and (len(parts) < self.maxn):
                parts.append(CuboidPrimitive())

    def _calculate_log_prior(self):
        """Prior for ShapeMaxN class.

        We assume a uniform prior over hypotheses.

        Returns:
            int: 0.0
        """
        return 0.0

    def copy(self):
        """Returns a (deep) copy of the ShapeMaxN instance

        Returns:
            ShapeMaxN: A (deep) copy of the instance
        """
        # NOTE that we are not copying params. This assumes that
        # params are not changing from hypothesis to hypothesis.
        self_copy = ShapeMaxN(forward_model=self.forward_model, maxn=self.maxn, params=self.params)
        parts_copy = deepcopy(self.parts)
        viewpoint_copy = deepcopy(self.viewpoint)
        self_copy.parts = parts_copy
        self_copy.viewpoint = viewpoint_copy
        return self_copy

if __name__ == "__main__":
    import mcmclib.proposal
    from i3d import i3d_proposal, vision_forward_model as vfm

    fwm = vfm.VisionForwardModel(render_size=(200, 200))
    max_part_count = 10
    h = ShapeMaxN(forward_model=fwm, viewpoint=[(np.sqrt(2.), -np.sqrt(2.), 2.0)],
                  params={'LL_VARIANCE': 0.0001}, maxn=max_part_count)

    """
    moves = {'shape_add_remove_part': shape_add_remove_part, 'shape_move_part': shape_move_part,
             'shape_move_part_local': shape_move_part_local, 'shape_change_part_size': shape_change_part_size,
             'shape_change_part_size_local': shape_change_part_size_local, 'shape_move_object': shape_move_object,
             'change_viewpoint': i3d_proposal.change_viewpoint_z}
    """    
    
    moves = {'shape_add_remove_part': shape_add_remove_part, 'shape_move_part_local': shape_move_part_local,
             'shape_change_part_size_local': shape_change_part_size_local,
             'change_viewpoint': i3d_proposal.change_viewpoint_z}

    params = {'MOVE_PART_VARIANCE': 0.00005,
              'MOVE_OBJECT_VARIANCE': 0.00005,
              'CHANGE_SIZE_VARIANCE': 0.00005,
              'CHANGE_VIEWPOINT_VARIANCE': 30.0,
              'MAX_PART_COUNT': max_part_count}

    proposal = mcmclib.proposal.RandomMixtureProposal(moves, params)

    # data = np.load('data/test1_single_view.npy')
    data = np.load('data/stimuli20150624_144833/o1_single_view.npy')

    # choose sampler
    thinning_period = 2000
    sampler_class = 'pt'
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
