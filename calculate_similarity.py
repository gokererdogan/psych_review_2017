"""
Inferring 3D Shape from 2D Images

This file contains the script for calculating predictions of our model.
WARNING: For pickle to run properly, i.e., import the necessary modules), run this script from the root Infer3DShape
folder.

Created on Feb 1, 2016

Goker Erdogan
https://github.com/gokererdogan/
"""

import numpy as np
import scipy.misc as spmisc

import i3d.vision_forward_model as vfm
from mcmclib.mcmc_run import MCMCRun


def calculate_probability_image_given_hypothesis(img, h, angle_increment=5):
    """
    Calculates the log probability of image given hypothesis, p(I|H) = \int p(I,theta|H) dtheta, marginalizing out
    viewpoint theta. We assume p(theta) is uniform.

    Parameters:
        img (numpy.array): image
        h (I3DHypothesis): shape hypothesis
        angle_increment (int): angle between views used to approximate the integral

    Returns:
        float: log probability of image given hypothesis, averaged over all views
        float: log probability of image given hypothesis for the best view
    """
    theta_count = int(360.0 / angle_increment)
    phi_count = int(180.0 / angle_increment)
    ll = np.zeros((phi_count, theta_count))
    for phi_i in range(phi_count):
        # update phi for all viewpoints
        for v in range(len(h.viewpoint)):
            r, theta, phi = h.viewpoint[v]
            phi = (phi + angle_increment) % 180
            h.viewpoint[v] = [r, theta, phi]

        for theta_i in range(theta_count):
            # update theta for all viewpoints
            for v in range(len(h.viewpoint)):
                r, theta, phi = h.viewpoint[v]
                theta = (theta + angle_increment) % 360
                h.viewpoint[v] = [r, theta, phi]

                h._log_ll = None
                log_ll = h.log_likelihood(img)
                ll[phi_i, theta_i] = log_ll

    ll = np.ravel(ll)
    return spmisc.logsumexp(ll) - np.log(theta_count * phi_count), np.max(ll)


def calculate_similarity_image_given_image(data1, samples):
    """
    Calculate similarity between images data1 and data2 given samples from p(H, theta|data2).
    Similarity between data1 and data2 is defined to be p(data1|data2) calculated from
    p(data1|data2) = \iint p(data1|H, theta) p(H|data2) p(theta) dH dtheta.
    samples are a list of samples from p(H, theta|data2). We assume p(theta) is uniform.

    Parameters:
        data1 (np.array):
        samples (list of I3DHypothesis):
        log_probs: log posterior probabilities of samples in ``samples``

    Returns:
        float: log p(data1|data2) calculated based on samples
        float: log p(data1|data2) calculated based on samples with p(data|H) calculated from only the best view.
    """
    # calculate p(H|data2) for each sample in samples
    sample_count = len(samples)
    logp_data1 = np.zeros(sample_count)  # log ll averaged over all views
    best_logp_data1 = np.zeros(sample_count)  # log ll from the best view
    for i, sample in enumerate(samples):
        print('.'),
        logp_data1[i], best_logp_data1[i] = calculate_probability_image_given_hypothesis(data1, sample)
    print

    p_avg = spmisc.logsumexp(logp_data1) - np.log(sample_count)
    p_best = spmisc.logsumexp(best_logp_data1) - np.log(sample_count)

    return p_avg, p_best


def read_samples(run_file, forward_model):
    """
    Load the samples from run file and restore their forward models.

    Parameters:
        run_file (string): filename of the pickled MCMCRun object
        forward_model (VisionForwardModel): forward model used for rendering

    Returns:
        (list of I3DHypothesis): Samples from the chain
        (list of I3DHypothesis): Samples with the highest posterior from the chain
    """
    run = MCMCRun.load(run_file)

    samples = run.samples.samples[5:]
    for sample in samples:
        sample.forward_model = forward_model

    best_samples = run.best_samples.samples
    for sample in best_samples:
        sample.forward_model = forward_model

    return samples, best_samples


if __name__ == "__main__":
    fwm = vfm.VisionForwardModel()

    data_folder = "./data"
    samples_folder = "./results"

    # we would like to calculate the similarity between two objects: target and comparison
    target_object = 'test1'
    comparison_object = 'test2'

    # load the images for target and comparison
    target_data = np.load('{0:s}/{1:s}.npy'.format(data_folder, target_object))
    comparison_data = np.load('{0:s}/{1:s}.npy'.format(data_folder, comparison_object))

    # load the samples from the chains for target
    target_run_file = "{0:s}/{1:s}.pkl".format(samples_folder, target_object)
    target_samples, target_best_samples = read_samples(target_run_file, fwm)

    # load the samples from the chains for comparison
    comp_run_file = "{0:s}/{1:s}.pkl".format(samples_folder, comparison_object)
    comp_samples, comp_best_samples = read_samples(comp_run_file, fwm)

    # calculate similarities
    p_comp_target, p_comp_target_best = calculate_similarity_image_given_image(comparison_data, target_samples)

    p_target_comp, p_target_comp_best = calculate_similarity_image_given_image(target_data, comp_samples)

    p_comp_target_MAP, p_comp_target_MAP_best = calculate_similarity_image_given_image(comparison_data,
                                                                                       target_best_samples)

    p_target_comp_MAP, p_target_comp_MAP_best = calculate_similarity_image_given_image(target_data, comp_best_samples)

    # calculate similarities
    # there are various ways you can calculate similarities given samples from the chains
    # you might want to see which one makes the best predictions for your experiment
    similarities = {
        'log P(C|T)': p_comp_target,  # probability of comparison given target
        'log P(T|C)': p_target_comp,  # probability of target given comparison
        'log (P(C|T) + P(T|C)) / 2': (p_comp_target + p_target_comp) / 2.0,  # average of the above two
        'log P_{MAP}(C|T)': p_comp_target_MAP,  # probability of comparison given target calculated from best samples
        'log P_{MAP}(T|C)': p_target_comp_MAP,  # probability of target given comparison calculated from best samples
        'log (P_{MAP}(C|T) + P_{MAP}(T|C)) / 2': (p_comp_target_MAP + p_target_comp_MAP) / 2.0,  # avg of the above two
        # the following are the same probabilities but calculated using only the best viewpoint
        # (rather than averaging over all possible viewpoints)
        'log P(C|T) (only best view)': p_comp_target_best,
        'log P(T|C) (only best view)': p_target_comp_best,
        'log P(C|T) + P(T|C)) / 2 (only best view)': (p_comp_target_best + p_target_comp_best) / 2.0,
        'log P_{MAP}(C|T) (only best view)': p_comp_target_MAP_best,
        'log P_{MAP}(T|C) (only best view)': p_target_comp_MAP_best,
        'log (P_{MAP}(C|T) + P_{MAP}(T|C)) / 2 (only best view)': (p_comp_target_MAP_best + p_target_comp_MAP_best) / 2.0,
    }

    print('\n'.join([k + ':' + str(v) for k, v in similarities.items()]))
