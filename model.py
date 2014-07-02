


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

import visualise


def calibrate_noise_model(benchmarks, all_node_results, setup, run_name=None, model_filename="models/noise-with-outliers.stan", 
    iterations=5000, warmup=4000, chains=1):
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
    N_nodes, N_benchmarks = sum([(each[2] == setup) for each in all_node_results.values()]), len(benchmarks)
    node_teff_measured = np.zeros((N_nodes, N_benchmarks))
    node_teff_uncertainty = np.zeros((N_nodes, N_benchmarks))

    # Match benchmarks in each node, then include their results to the array.
    i = 0
    additive_variance = np.zeros((N_nodes, N_benchmarks))
    node_names = []
    for node, values in all_node_results.iteritems():
        node_results, node_flags, node_setup = values
        if node_setup != setup: continue

        # Match by name
        for j, benchmark in enumerate(benchmarks):
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
                node_teff_uncertainty[i, j] = node_results[match_by_object]["e_TEFF"]
            else:
                additive_variance[i, j] = 10000000000.

        node_names.append(node)
        i += 1

    # Prepare the data in a cleaner format.
    data = {
        "non_spec_teff_measured": benchmarks["TEFF"],
        "non_spec_teff_sigma": benchmarks["u_TEFF"],
        "N_nodes": N_nodes,
        "N_benchmarks": N_benchmarks,
        "node_teff_measured": node_teff_measured,
        "node_teff_uncertainty": node_teff_uncertainty,
        "additive_variance": additive_variance
    }

    logging.info("Optimizing...")
    op = model.optimizing(data=data)
    logging.info("Optimized Values: \n{0}".format(op["par"]))

    logging.info("Fitting...")
    calibrated_model = model.sampling(data=data, pars=op["par"], iter=iterations, warmup=warmup, chains=chains)    
    
    return (calibrated_model, node_names, data)


def homogenise(data, model):
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
    calibrated_data_weighted = np.zeros(N_samples)
    for j in xrange(N_samples):
        weights = np.matrix(covariance[j]).I * E.T # Phone Home
        weights /= sum(weights)
        calibrated_data_weighted[j] = (np.random.multivariate_normal(data[valid_data_indices], covariance[j]) * weights)[0,0]

    return np.median(calibrated_data_weighted), np.std(calibrated_data_weighted)


def cross_validate(benchmarks, all_node_results, setup, **kwargs):
    """
    Remove a benchmark from the sample. Calibrate the model. Use the calibrated model to homogenise node
    measurements for the benchmark star. Use this to judge the predictive power of the model.
    """

    # Measurements & Uncertainty.
    measurements = np.zeros((len(benchmarks), 2))
    for i, benchmark in enumerate(benchmarks):
        # Create a subset with this benchmark removed.
        benchmark_subset = np.delete(benchmarks, i)

        # Calibrate the model on our subset.
        calibrated_model, node_names, model_data = calibrate_noise_model(benchmark_subset, all_node_results, setup, **kwargs)

        # Get the measurements for this benchmark.
        benchmark_measurements = np.zeros(len(node_names))
        for j, node_name in enumerate(node_names):
            node_data = all_node_results[node_name][0]
            index = np.where((node_data["OBJECT"] == benchmark["OBJECT"]) * (node_data["SETUP"] == setup))[0]
            assert len(index) == 1
            benchmark_measurements[j] = node_data["TEFF"][index[0]]

        # Homogenise the measurements for this benchmark.
        measurements[i] = homogenise(benchmark_measurements, calibrated_model, **kwargs)

    return measurements

if __name__ == "__main__":

    setup, model_name = "UVES", "preliminary_test-{0}".format(md5(os.path.basename(ctime()).encode("utf-8")).hexdigest())
    model, node_names, data = calibrate_noise_model(benchmark_truths, all_node_results, setup)
    
    # Draw the node uncertainties
    fig1 = visualise.node_uncertainties(model, node_names)

    # Do a cross-validation of all the benchmark stars
    results = cross_validate(benchmark_truths, all_node_results, setup)

    # Do some post-processing of the cross-validation results
    fig2 = visualise.boxplots(benchmark_truths, data, results[:,0], results[:,1])
    fig3 = visualise.node_benchmark_measurements(benchmark_truths, data, setup, node_names, results[:,0], results[:,1])

