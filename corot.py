#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Homogenisation model for GES/CoRoT data. """

__author__ = "Andy Casey (arc@ast.cam.ac.uk)"

import cPickle as pickle
import logging
import os
from hashlib import md5
from time import ctime

import numpy as np
from pystan import StanModel


def build_data_dict(benchmarks, all_node_results, non_finite_variance=10e9):
    """
    Prepare a dictionary containing the data required for the model.
    """

    # The data needs to be in a cleaned format.
    N_nodes, N_benchmarks = len(all_node_results), len(benchmarks)
    node_teff_measured = np.zeros((N_nodes, N_benchmarks))
    node_teff_uncertainty = np.zeros((N_nodes, N_benchmarks))

    # Match benchmarks in each node, then include their results to the array.
    i = 0
    additive_variance = np.zeros((N_nodes, N_benchmarks))
    node_names = []
    for node, values in all_node_results.iteritems():
        node_results, node_flags, node_setup = values
        
        # Match by name
        for j, benchmark in enumerate(benchmarks):
            match_by_object = np.where((node_results["OBJECT"] == benchmark["OBJECT"]) * (node_results["SETUP"] == node_setup))[0]
            if len(match_by_object) == 0:
                logging.warn("Benchmark {0} not found in {1} results!".format(benchmark["OBJECT"], node))
                continue
            elif len(match_by_object) > 1:
                raise IndexError("found multiple benchmarks {0} for setup {1} by node {2}".format(benchmark["OBJECT"], node_setup, node))

            # If the measurement is negative or non-finite, then add a lot of variance to this measurement.
            measurement = node_results[match_by_object]["TEFF"]
            if np.isfinite(measurement) and measurement > 0:
                node_teff_measured[i, j] = measurement
                node_teff_uncertainty[i, j] = node_results[match_by_object]["e_TEFF"]
            else:
                additive_variance[i, j] = non_finite_variance
        i += 1
        node_names.append(node)

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
    return data, node_names


def calibrate_noise_model(benchmarks, all_node_results, run_name=None,
    model_filename="models/noise-with-outliers.stan", iterations=9000, warmup=8000, chains=1):
    """
    Run the given noise model for the benchmark stars.
    """

    if run_name is None:
        run_name = "unnamed"

    else:
        # If a name has been given, use a timestamp too.
        run_name = "-".join([run_name, format(md5(ctime().encode("utf-8")).hexdigest())])

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

    data, node_names = build_data_dict(benchmarks, all_node_results)

    logging.info("Optimizing...")
    op = model.optimizing(data=data)
    logging.info("Optimized Values: \n{0}".format(op["par"]))

    logging.info("Fitting...")
    calibrated_model = model.sampling(data=data, pars=op["par"], iter=iterations, warmup=warmup, chains=chains)    
    
    # Add the node names into the data dict.
    calibrated_model.data["node_names"] = node_names

    return calibrated_model


def homogenise(data, model):
    """
    Homogenise the data for stars given the calibrated model.

    data : ndarray of size (N, ) where N is number of nodes.
    model : fitted model
    """

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

    return [np.median(calibrated_data_weighted), np.std(calibrated_data_weighted), N_valid_measurements]


def homogenise_all(all_node_results, model):
    """
    Homogenise the data from all nodes, for all stars.
    """

    # Need to identify stars that have this setup.
    first_node = all_node_results.keys()[0]
    setup = all_node_results[first_node][2]
    cnames = all_node_results[first_node][0][(all_node_results[first_node][0]["SETUP"] == setup)]["CNAME"]
    filenames = all_node_results[first_node][0][(all_node_results[first_node][0]["SETUP"] == setup)]["FILENAME"]
    snr = all_node_results[first_node][0][(all_node_results[first_node][0]["SETUP"] == setup)]["SNR"]
    num_stars = len(cnames)

    # Go through each star, find it in all other nodes.
    homogenised_results = np.zeros((num_stars, 3))
    for i, (cname, filename) in enumerate(zip(cnames, filenames)):

        # Match by CNAME in all the nodes.
        data = np.zeros(len(all_node_results))
        for j, (node, node_results) in enumerate(all_node_results.iteritems()):

            matches = (node_results[0]["CNAME"] == cname) * (node_results[0]["SETUP"] == setup)
            if sum(matches) > 1:
                matches *= (node_results[0]["FILENAME"] == filename)

            assert sum(matches) > 0, "Could not match {0} for setup {1} in node results {2}".format(
                cname, setup, node)
            assert 2 > sum(matches), "There were multiple matches for CNAME, FILENAME ({0}, {1})"\
                "in node results {2}".format(cname, filename, node)
            data[j] = node_results[0][matches]["TEFF"]

        # Homogenise the data.            
        homogenised_results[i] = homogenise(data, model)

    # Return a table with all the homogenised values.
    return (cnames, snr, homogenised_results)
 

def cross_validate(benchmarks, all_node_results, **kwargs):
    """
    Remove a benchmark from the sample. Calibrate the model. Use the calibrated model to homogenise node
    measurements for the benchmark star. Use this to judge the predictive power of the model.
    """

    # Measurements & Uncertainty.
    results = np.zeros((len(benchmarks), 2))
    for i, benchmark in enumerate(benchmarks):
        # Create a subset with this benchmark removed.
        benchmark_subset = np.delete(benchmarks, i)

        # Calibrate the model on our subset.
        calibrated_model = calibrate_noise_model(benchmark_subset, all_node_results, **kwargs)

        # Get the measurements for this benchmark.
        benchmark_measurements = np.zeros(len(all_node_results))
        for j, (node_name, node_results) in enumerate(all_node_results.iteritems()):
            node_data, node_tech_flags, node_setup = node_results
            index = np.where((node_data["OBJECT"] == benchmark["OBJECT"]) * (node_data["SETUP"] == node_setup))[0]
            assert len(index) == 1
            benchmark_measurements[j] = node_data["TEFF"][index[0]]

        # Homogenise the measurements for this benchmark.
        results[i] = homogenise(benchmark_measurements, calibrated_model)

    measurements, uncertainties = results[:,0], results[:,1]
    return (measurements, uncertainties)



