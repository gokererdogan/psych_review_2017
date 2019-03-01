This repository contains the code for the paper "Erdogan G., Jacobs R. A. (2017) Visual Shape Perception as Bayesian Inference of 3D Object-centered Shape Representations. Psychological Review".

See `run_bdaooss_experiment.py` for a script that runs the MCMC chain to sample 3D shape hypotheses given an image.

To calculate the predictions of our model in the paper, see `calculate_similarity.py` script. Note you will need the sample produced by the MCMC chain to calculate similarities.
