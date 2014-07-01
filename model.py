


import cPickle as pickle
import logging
import os
from hashlib import md5
from textwrap import dedent
from time import ctime

import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
from pystan import StanModel

from load_data import *


def calibrate_noise_model(all_node_results, setup, run_name=None, model_filename="models/noise-with-outliers.stan", 
    iterations=20000, warmup=13000, chains=1):
    """
    Run the given noise model for the benchmark stars.
    """

    if run_name is None:
        run_name = "unnamed"

    else:
        # If a name has been given, use a timestamp too.
        run_name = "-".join([run_name, format(md5(ctime().encode("utf-8")).hexdigest())])

    # Include the setup name in the run name.
    run_name = "-".join([setup.lower(), run_name])

    # Check for a compiled version of this model.
    basename, ext = os.path.splitext(model_filename)
    if os.path.exists(basename + ".pkl"):
        # There's a compiled version. Use that.
        model_filename = basename + ".pkl"
        logging.info("Using pre-compiled model {0}".format(model_filename))
        with open(model_filename, "rb") as fp:
            model = pickle.load(fp)

    else:
        # Compilation required.
        model = StanModel(model_filename)
        pickled_model_filename = basename + ".pkl"
        logging.info("Pickling compiled model to {0}".format(pickled_model_filename))
        with open(pickled_model_filename, "wb") as fp:
            pickle.dump(model, fp)

    # The data needs to be in a cleaned format.
    N_nodes, N_benchmarks = sum([(each[2] == setup) for each in all_node_results.values()]), len(benchmark_truths)
    node_teff_measured = np.zeros((N_nodes, N_benchmarks))

    # Match benchmarks in each node, then include their results to the array.
    i = 0
    additive_variance = np.zeros((N_nodes, N_benchmarks))
    node_names = []
    for node, values in all_node_results.iteritems():
        node_results, node_flags, node_setup = values
        if node_setup != setup: continue

        # Match by name
        for j, benchmark in enumerate(benchmark_truths):
            match_by_object = np.where((node_results["OBJECT"] == benchmark["OBJECT"]) * (node_results["SETUP"] == setup))[0]
            if len(match_by_object) == 0:
                logging.warn("Benchmark {0} not found in {1} results!".format(benchmark["OBJECT"], node))
                continue
            elif len(match_by_object) > 1:
                raise IndexError("found multiple benchmarks {0} for setup {1} by node {2}".format(benchmark["OBJECT"], setup, node))

            # If the measurement is negative or non-finite, then add a lot of variance to this measurement.
            measurement = node_results[match_by_object]["TEFF"]
            if np.isfinite(measurement) and measurement > 0:
                node_teff_measured[i, j] = measurement
            else:
                additive_variance[i, j] = 10000000000.

        node_names.append(node)
        i += 1

    # Prepare the data in a cleaner format.
    data = {
        "non_spec_teff_measured": benchmark_truths["TEFF"],
        "non_spec_teff_sigma": benchmark_truths["u_TEFF"],
        "N_nodes": N_nodes,
        "N_benchmarks": N_benchmarks,
        "node_teff_measured": node_teff_measured,
        "additive_variance": additive_variance
    }

    logging.info("Optimizing...")
    op = model.optimizing(data=data)
    logging.info("Optimized Values: \n{0}".format(op["par"]))

    logging.info("Fitting...")
    calibrated_model = model.sampling(data=data, pars=op["par"], iter=iterations, warmup=warmup, chains=chains)    
    
    return (calibrated_model, node_names, data)


def homogenise(data, model, max_mixtures=1, n_iter=1000, **kwargs):
    """
    Homogenise the data for stars given the calibrated model.

    data : ndarray of size (N, ) where N is number of nodes.
    model : fitted model
    """

    assert max_mixtures >= 1

    # Use mu from data, and draw uncertainty from each sample.
    extracted_samples = model.extract(permuted=True)
    N_samples, N_nodes = extracted_samples["var_node"].shape
    assert len(data) == N_nodes

    # Get number of valid measurements.
    valid_data_indices = (data > 0) * np.isfinite(data)
    N_valid_measurements = np.sum(valid_data_indices)
    if N_valid_measurements == 0:
        raise ValueError("no valid measurements found: 0 >= all data!")

    # Fill up the covariance matrix.
    covariance = np.zeros((N_samples, N_valid_measurements, N_valid_measurements))
    for j in xrange(N_samples):
        covariance[j, :, :] += extracted_samples["var_intrinsic"][j]

    for i, measurement in enumerate(data[valid_data_indices]):
        for j in xrange(N_samples):
            covariance[j, i, i] += extracted_samples["var_node"][j, i]

    # Sample.
    E = np.matrix(np.ones(N_valid_measurements))
    original_data_weighted = []
    calibrated_data_weighted = []
    calibrated_data_unweighted = np.zeros((N_samples, N_valid_measurements))
    for j in xrange(N_samples):
        weights = np.matrix(covariance[j]).I * E.T # Phone Home
        weights /= sum(weights)

        original_data_weighted.append((data[valid_data_indices] * weights)[0,0])
        calibrated_data_weighted.append((samples[j, :] * weights)[0,0])
        calibrated_data_unweighted[j, :] = np.random.multivariate_normal(data[valid_data_indices], covariance[j])
        
    calibrated_data_unweighted = calibrated_data_unweighted.flatten()

    """
    # With our built-up array, fit the data with multiple Gaussian mixture models.
    aics = []
    components = range(1, 1 + max_mixtures)
    for each in components:
        mixture_model = mixture.GMM(each, covariance_type='full', min_covar=1e-5, n_iter=n_iter, **kwargs)
        mixture_model = mixture_model.fit(calibrated_data_unweighted)

        # Calculate AIC
        aics.append(mixture_model.aic(calibrated_data_unweighted))
    """
    print("ORIGINAL     (WEIGHTED): {0:.0f} +/- {1:.0f}".format(np.median(original_data_weighted), np.std(original_data_weighted)))
    print("CALIBRATED (UNWEIGHTED): {0:.0f} +/- {1:.0f}".format(np.median(calibrated_data_unweighted), np.std(calibrated_data_unweighted)))
    print("CALIBRATED   (WEIGHTED): {0:.0f} +/- {1:.0f}".format(np.median(calibrated_data_weighted), np.std(calibrated_data_weighted)))

    chosen_one = calibrated_data_weighted
    return np.median(chosen_one), np.std(chosen_one)


if __name__ == "__main__":


    # Create a time-based string
    model_name = "preliminary_test-{0}".format(md5(os.path.basename(ctime()).encode("utf-8")).hexdigest())
    model, node_names, data = calibrate_noise_model(all_node_results, "UVES")
    
    # Draw histograms showing the uncertainty for each node, and plots showing the data and inferred uncertainties.
    # sort by median:
    samples = model.extract(permuted=True)
    fig, ax = plt.subplots()
    ax.set_color_cycle(["#348ABD", "#7A68A6", "#A60628", "#467821", "#CF4457", "#188487", "#E24A33", "#000000"])
    bins = np.linspace(0, 1000, samples["var_node"].shape[0]**0.5)
    sorted_indices = np.argsort(np.median(samples["var_node"], axis=0))

    for index in sorted_indices:

        node_name = node_names[index]

        y = np.histogram(samples["var_node"][:, index]**0.5, bins, normed=True)
        ax.plot(bins[1:], y[0], label=node_name, lw=2)

    ax.legend(loc=1, frameon=False)
    ax.set_xlabel("$\sigma\left({T_{\\rm eff}}\\right)$")
    ax.set_ylabel("Normalised count")

    fig.savefig("plots/node-uncertainty-with-outlier-mixture.pdf")

    # Recommend values for other measurements.
    # When recommending other values, each star needs to be considered separately with a mixture model.



    # Plot recommended values against all the original node values (with calculated uncertainties)

    raise a

    #recommended_parameters = np.array([homogenise_mode_weighted(model, observations[:, j], dimensions)["teff"][0] for j in xrange(23)]).reshape(1, 23)

    #observations[observations <= 0] = np.nan
    #fig = plot.boxplots(benchmarks, observations, ("TEFF",), recommended_values=recommended_parameters)

    #fig[0].savefig(model_name + "-recommended.jpg")

    #recommended_parameters_alpha = [homogenise_mode_alpha(model, observations[:, i, :], dimensions) for i in xrange(23)]

