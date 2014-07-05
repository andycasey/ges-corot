#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Homogenise GIRAFFE results for the GES/CoRoT project. """

__author__ = "Andy Casey (arc@ast.cam.ac.uk)"

import os
from hashlib import md5
from time import ctime

import corot
import visualise
from load_data import benchmark_truths, giraffe_results

# Run name.
run_name = "giraffe-{0}".format(md5(os.path.basename(ctime()).encode("utf-8")).hexdigest())
run_name = None

# Draw all the node measurements first
fig_node_all_measurements = visualise.node_all_measurements(giraffe_results)
fig_node_all_measurements.savefig("plots/giraffe-all-measurements.pdf")

# Calibrate the model
model = corot.calibrate_noise_model(benchmark_truths, giraffe_results, run_name=run_name)

# Draw the node uncertainties
fig_node_uncertainties = visualise.inferred_node_uncertainties(model)
fig_node_uncertainties.savefig("plots/giraffe-inferred-uncertainties.pdf")

# Do a cross-validation of all the benchmark stars
cv_measurements, cv_uncertainties = corot.cross_validate(benchmark_truths, giraffe_results)

# Do some post-processing of the cross-validation results
fig_boxplots = visualise.boxplots(benchmark_truths, model, cv_measurements, cv_uncertainties)
fig_boxplots.savefig("plots/giraffe-benchmark-boxplot.pdf")
fig_node_benchmark_measurements = visualise.node_benchmark_measurements(benchmark_truths, model,
    cv_measurements, cv_uncertainties)
fig_node_benchmark_measurements.savefig("plots/giraffe-benchmark-measurements.pdf")

# Homogenise all the data.
giraffe_cnames, giraffe_snr, giraffe_homogenised_results = corot.homogenise_all(giraffe_results, model)

# For stars with multiple measurements, plot the results.
fig_multiple_measurements = visualise.repeat_measurements(giraffe_cnames, giraffe_snr, giraffe_homogenised_results)
fig_multiple_measurements.savefig("plots/giraffe-multiple-measurements.pdf")

# Visualise the homogenised results in context of the initial node results as a corner plot.
fig_uncertainties_vs_snr = visualise.snr_vs_uncertainties(giraffe_snr, giraffe_homogenised_results)
fig_uncertainties_vs_snr.savefig("plots/giraffe-snr-vs-uncertainties.pdf")

# Create a GES/CoRoT-standardised FITS table with the homogenised results.
hdulist = corot.update_table("results/homogenised_parameters.fits",
    giraffe_cnames, giraffe_homogenised_results, setup="GIRAFFE")
hdulist.writeto("results/homogenised_parameters.fits", clobber=True)

# Visualise the homogenised results in context of the initial node results as a corner plot.
giraffe_results_incl_ensemble = giraffe_results.copy()
giraffe_results_incl_ensemble["ENSEMBLE RESULT"] = (load_node_data("results/homogenised_parameters.fits", 1), "NONE", "GIRAFFE")
fig_node_all_measurements = visualise.node_all_measurements(giraffe_results_incl_ensemble)
fig_node_all_measurements.savefig("plots/giraffe-all-measurements-incl-ensemble.pdf")
