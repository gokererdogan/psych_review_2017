"""
Inferring 3D Shape from 2D Images

This file contains the experiment script for running the chains with BDAoOSSShape hypothesis on different inputs
and with different parameters.

Created on Oct 12, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import experiment as exp

from experiment.run_chain import *


def run_bdaooss_experiment(**kwargs):
    """This method runs the chain with a BDAoOSSShape hypothesis and given parameters.

    This method is intended to be used in an Experiment instance. This method prepares the necessary data and
    calls `run_chain`.

    Parameters:
        kwargs (dict): Keyword arguments are as follows
            input_file (str): mame of the data file containing the observed image
            data_folder (str): folder containing the data files
            results_folder (str):
            sampler (str): see `run_chain` function
            offscreen_rendering (bool): If True, renders offscreen.
            max_depth (int): maximum depth of the hypothesis trees
            ll_variance (float): variance of the Gaussian likelihood
            max_pixel_value (float): maximum pixel intensity value
            change_size_variance (float): variance for the change part size move
            change_viewpoint_variance (float): variance for the change viewpoint move
            burn_in (int): see `run_chain` function
            sample_count (int): see `run_chain` function
            best_sample_count (int): see `run_chain` function
            thinning_period (int): see `run_chain` function
            report_period (int): see `run_chain` function
            temperatures (list): see `run_chain` function

    Returns:
        dict: run results
    """
    try:
        input_file = kwargs['input_file']
        results_folder = kwargs['results_folder']
        data_folder = kwargs['data_folder']
        sampler = kwargs['sampler']
        offscreen_rendering = kwargs['offscreen_rendering']
        ll_variance = kwargs['ll_variance']
        max_pixel_value = kwargs['max_pixel_value']
        max_depth = None
        if 'max_depth' in kwargs:
            max_depth = kwargs['max_depth']
        change_size_variance = kwargs['change_size_variance']
        change_viewpoint_variance = kwargs['change_viewpoint_variance']
        burn_in = kwargs['burn_in']
        sample_count = kwargs['sample_count']
        best_sample_count = kwargs['best_sample_count']
        thinning_period = kwargs['thinning_period']
        report_period = kwargs['report_period']
        temperatures = None
        if 'temperatures' in kwargs:
            temperatures = kwargs['temperatures']
    except KeyError as e:
        raise ValueError("All experiment parameters should be provided. Missing parameter {0:s}".format(e.message))

    import numpy as np

    import mcmclib.proposal as proposal
    from i3d import i3d_proposal
    from i3d import vision_forward_model as vfm

    # read the data file
    viewpoint = [[np.sqrt(8.0), -45.0, 45.0]]
    data = np.load("{0:s}/{1:s}.npy".format(data_folder, input_file))
    custom_lighting = True

    render_size = data.shape[1:]
    fwm = vfm.VisionForwardModel(render_size=render_size, custom_lighting=custom_lighting,
                                 offscreen_rendering=offscreen_rendering)

    shape_params = {'LL_VARIANCE': ll_variance, 'MAX_PIXEL_VALUE': max_pixel_value}

    kernel_params = {'CHANGE_SIZE_VARIANCE': change_size_variance,
                     'CHANGE_VIEWPOINT_VARIANCE': change_viewpoint_variance}

    from bdaooss import bdaooss_shape as bdaooss

    moves = {'change_viewpoint': i3d_proposal.change_viewpoint_z,
             'bdaooss_add_remove_part': bdaooss.bdaooss_add_remove_part,
             'bdaooss_change_part_size_local': bdaooss.bdaooss_change_part_size_local,
             'bdaooss_change_part_dock_face': bdaooss.bdaooss_change_part_dock_face}

    if max_depth is None:
        h = bdaooss.BDAoOSSShape(forward_model=fwm, viewpoint=viewpoint, params=shape_params)

    else:
        from bdaooss import bdaooss_shape_maxd as bdaooss_maxd
        h = bdaooss_maxd.BDAoOSSShapeMaxD(forward_model=fwm, max_depth=max_depth, viewpoint=viewpoint,
                                          params=shape_params)
        kernel_params['MAX_DEPTH'] = max_depth

    # form the proposal
    kernel = proposal.RandomMixtureProposal(moves=moves, params=kernel_params)

    results = run_chain(name=input_file, sampler=sampler, initial_h=h, data=data, kernel=kernel, burn_in=burn_in,
                        thinning_period=thinning_period, sample_count=sample_count, best_sample_count=best_sample_count,
                        report_period=report_period, results_folder=results_folder, temperatures=temperatures)

    return results


if __name__ == "__main__":
    # these are the names of the images you would like to infer the 3D shape for. note these are in numpy array format.
    # see some examples in the data folder
    input_files = ['test1', 'test2']

    # maximum pixel value in images rendered by VTK. If you change lighting in the scene, this will change.
    MAX_PIXEL_VALUE = 177.0  # this is usually 256.0 but in our case because of the lighting in our renders, it is lower

    # the experiment class allows you to run multiple chains (either in parallel or consecutively) and save all the
    # results. You don't need to use it but it will make it easier to run the chain for many images
    # below are the parameters used for the results in the paper. You may need to tune them for your stimuli.
    experiment = exp.Experiment(name="TestExperiment", experiment_method=run_bdaooss_experiment,
                                sampler=['mh'],
                                input_file=input_files,
                                results_folder='./results',
                                data_folder='./data/',
                                # set to False if you'd like to see hypotheses rendered in real-time.
                                # note this will slow down the chain significantly.
                                offscreen_rendering=True,
                                max_depth=5,  # maximum depth for shape trees
                                max_pixel_value=MAX_PIXEL_VALUE,
                                ll_variance=[0.0001],  # likelihood variance
                                change_size_variance=[0.00005],  # variance for change size move
                                change_viewpoint_variance=[10.0],  # variance for change viewpoint move
                                # chain parameters
                                burn_in=5000, sample_count=10, best_sample_count=10, thinning_period=10000,
                                report_period=1000)

    # run the experiment. if you'd like to run chains for multiple input files at the same time taking advantage of
    # multiprocessing, set parallel=True. num_processes determines the number of processes running in parallel.
    experiment.run(parallel=False, num_processes=-1)

    # save the results
    # this will create
    #     - a folder for each input file with images of samples from the chain
    #         - s1, s2, s3, ... are the samples
    #         - b1, b2, b3, ... are the samples with highest posterior probabilities
    #     - a pickled file of samples for each input file
    #     - a CSV file of experiment results
    #     - a pickled file of experiment results
    print(experiment.results)
    experiment.save('./results')
    experiment.append_csv('./results/TestExperiment.csv')
