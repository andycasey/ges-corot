#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Homogenise UVES results for the GES/CoRoT project. """

__author__ = "Andy Casey (arc@ast.cam.ac.uk)"

import os
import cPickle as pickle
from collections import OrderedDict
from hashlib import md5
from time import ctime

import corot
import visualise
from load_data import benchmark_truths, load_node_data, uves_results

# Run name.
run_name = "uves-{0}".format(md5(os.path.basename(ctime()).encode("utf-8")).hexdigest())
run_name = None

# Draw all the node measurements first
fig_node_all_measurements = visualise.node_all_measurements(uves_results)
fig_node_all_measurements.savefig("plots/uves-all-measurements.pdf")

# Calibrate the model
model = corot.calibrate_noise_model(benchmark_truths, uves_results, run_name=run_name)

# Draw the node uncertainties
fig_node_uncertainties = visualise.inferred_node_uncertainties(model)
fig_node_uncertainties.savefig("plots/uves-inferred-uncertainties.pdf")

# Do a cross-validation of all the benchmark stars
cv_measurements, cv_uncertainties = corot.cross_validate(benchmark_truths, uves_results)

# Do some post-processing of the cross-validation results
fig_boxplots = visualise.boxplots(benchmark_truths, model, cv_measurements, cv_uncertainties)
fig_boxplots.savefig("plots/uves-benchmark-boxplot.pdf")
fig_node_benchmark_measurements = visualise.node_benchmark_measurements(benchmark_truths, model,
    cv_measurements, cv_uncertainties)
fig_node_benchmark_measurements.savefig("plots/uves-benchmark-measurements.pdf")

# Homogenise all the data.
uves_cnames, uves_snr, uves_homogenised_results = corot.homogenise_all(uves_results, model)

# Plot the uncertainties as a function of S/N.
fig_uncertainties_vs_snr = visualise.snr_vs_uncertainties(uves_snr, uves_homogenised_results)
fig_uncertainties_vs_snr.savefig("plots/uves-snr-vs-uncertainties.pdf")

# Create a GES/CoRoT-standardised FITS table with the homogenised results.
hdulist = corot.update_table("results/homogenised_parameters.fits",
    uves_cnames, uves_homogenised_results, setup="UVES")
hdulist.writeto("results/homogenised_parameters.fits", clobber=True)

# Visualise the homogenised results in context of the initial node results as a corner plot.
uves_results_incl_ensemble = OrderedDict([
    ("ENSEMBLE RESULT", (load_node_data("results/homogenised_parameters.fits", 1), "NONE", "UVES"))])
uves_results_incl_ensemble.update(uves_results)
fig_node_all_measurements = visualise.node_all_measurements(uves_results_incl_ensemble)
fig_node_all_measurements.savefig("plots/uves-all-measurements-incl-ensemble.pdf")

# Pickle the results.
with open("results/uves{0}.pkl".format("-" + run_name if run_name is not None else ""), "wb") as fp:
    pickle.dump({
        "model": model,
        "cv_measurements": cv_measurements,
        "cv_uncertainties": cv_uncertainties,
        "uves_cnames": uves_cnames,
        "uves_snr": uves_snr,
        "uves_homogenised_results": uves_homogenised_results
    }, fp, -1)
