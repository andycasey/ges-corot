#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Load and validate GES/CoRoT node measurements. """

__author__ = "Andy Casey (arc@ast.cam.ac.uk)"

import logging
from collections import OrderedDict

import numpy as np
import pyfits

__all__ = ["all_node_results", "benchmark_truths", "initial_parameters", "validate_all",
    "uves_results", "giraffe_results"]

def load_benchmarks(filename="benchmarks.txt"):
    """ Loads the Gaia benchmarks and expected stellar parameters as
    a record array """

    data = np.loadtxt(filename, dtype=str)
    max_filenames_len, max_cname_len, max_object_len = (max(map(len, data[:, i])) for i in xrange(3))

    benchmarks = np.core.records.fromarrays(data.T,
        names=["FILENAME", "CNAME", "Object", "TEFF",
            "u_TEFF", "LOGG", "u_LOGG", "FEH"],
        formats=[
            "|S{0:.0f}".format(max_filenames_len),
            "|S{0:.0f}".format(max_cname_len),
            "|S{0:.0f}".format(max_object_len),
            "f8", "f8", "f8", "f8", "f8"])

    return benchmarks


def load_node_data(filename, ext):
    data = pyfits.open(filename)[ext].data

    # Sort by CNAME
    indices = np.argsort(data["CNAME"])
    return data[indices]


def validate(node, all_node_results):
    """
    Check that the new node parameters are not the original input parameters.
    """

    all_ok = True

    sort_a = np.argsort(initial_parameters["CNAME"])
    sort_b = np.argsort(all_node_results[node][0]["CNAME"])
    setup = all_node_results[node][2]

    same_temperatures = (initial_parameters[sort_a]["TEFF"] == all_node_results[node][0][sort_b]["TEFF"]) * (initial_parameters[sort_a]["SETUP"] == setup)
    if np.any(same_temperatures):
        logging.warn("Node {0} has {1} TEFF measurements that match the input values: {2}".format(node, np.sum(same_temperatures),
            ", ".join(list(initial_parameters[sort_a]["CNAME"][same_temperatures]))))
        all_ok = False

    same_temperature_uncertainties = (initial_parameters[sort_a]["e_TEFF"] == all_node_results[node][0][sort_b]["e_TEFF"]) * (initial_parameters[sort_a]["SETUP"] == setup)
    if np.any(same_temperature_uncertainties):
        logging.warn("Node {0} has {1} TEFF uncertainties that match the input values: {2}".format(node, np.sum(same_temperature_uncertainties),
            ", ".join(list(initial_parameters[sort_a]["CNAME"][same_temperature_uncertainties]))))
        all_ok = False

    final = all_node_results[node][0][sort_b]
    initial = initial_parameters[sort_a]

    logg_difference = (np.abs(final["LOGG"] - initial["LOGG"]) > 0) * (initial["SETUP"] == setup)
    if np.any(logg_difference):
        logging.warn("Node {0} has not fixed LOGG measurements for {1} stars (mean difference: {2:+.2f}, std. dev: {3:.2f}): {4}".format(node,
            np.sum(logg_difference), np.mean(final["LOGG"][logg_difference] - initial["LOGG"][logg_difference]), 
            np.std(final["LOGG"][logg_difference] - initial["LOGG"][logg_difference]),
            ", ".join(list(final["CNAME"][logg_difference]))))
        all_ok = False

    return all_ok

# Functions defined. Load the data.
uves_results = OrderedDict([
    ("BOLOGNA", (load_node_data("data/bologna/Bologna_GES_CoRoTv2/GES_CoRoT_25May2014_Bologna_NEW-cleaned.fits", 1), "INVALID", "UVES")), # Those with TECH flags 10001A, 10002A, 10003A
    ("LUMBA (H-$\\alpha$)", (load_node_data("data/lumba/25june/GES_CoRoT_25May2014_LUMBA_HalphaTeff-cleaned.fits", 1), "INVALID", "UVES")), # 11110A, 10023A|11110A, 'double line or emission', 'HR15 not RV-corrected correctly'
    ("VILNIUS", (load_node_data("data/vilnius/GES_Corot_iteration1_Vilnius.v11-corrected.fits", 1), "NONE_MARKED_AS_INVALID", "UVES")), # None INVALID
    ("EPINARBO", (load_node_data("data/epinarbo/Results_CoRoT_Epinarbo_18June2014/GES_CoRoT_Epinarbo_18June2014-v2.fits", 1), "NONE_MARKED_AS_INVALID", "UVES")), # None INVALID
    ("LUMBA (NLTE)", (load_node_data("data/lumba-nlte/GES_iDR3_WG11_LUMBA-CoRoT.fits", 1), "NONE_MARKED_AS_INVALID", "UVES")), # None INVALID
    ("SOUSA", (load_node_data("data/sousa/Results_line_ratio_sousa-cleaned.fits", 1), "INVALID", "UVES")), # Those outside of their regime (as determined by them, not other nodes)
    ("PHOTOMETRY", (load_node_data("data/photometry/GES_CoRoT_PhotTeff.fits", 1), "NONE_MARKED_AS_INVALID", "UVES")), # None INVALID
    ("IACAIP", (load_node_data("data/iacaip/GES_CoRoT_IACAIP_v0.fits", 1), "NONE_MARKED_AS_INVALID", "UVES")),  # None INVALID
    ("ARCETRI", (load_node_data("data/arcetri/GES_CoRoT_Arcetri_iter1.fits", 1), "INVALID", "UVES")), # TECH flag of 1 or 2
])

giraffe_results = OrderedDict([
    ("LUMBA (H-$\\alpha$)", (load_node_data("data/lumba/25june/GES_CoRoT_25May2014_LUMBA_HalphaTeff-cleaned.fits", 1), "INVALID", "GIRAFFE")), # 11110A, 10023A|11110A, 'double line or emission', 'HR15 not RV-corrected correctly'
    ("LUMBA (EXCITATION)", (load_node_data("data/lumba/25june/GES_CoRoT_25May2014_LUMBA_ExcTeff-cleaned.fits", 1), "INVALID", "GIRAFFE")), # 10023A, 'Poor fit - binary?', 'HR15N not RV correctly'
    ("EPINARBO", (load_node_data("data/epinarbo/Results_CoRoT_Epinarbo_18June2014/GES_CoRoT_Epinarbo_18June2014-v2.fits", 1), "NONE_MARKED_AS_INVALID", "GIRAFFE")), # None INVALID
    ("SOUSA", (load_node_data("data/sousa/Results_line_ratio_sousa-cleaned.fits", 1), "INVALID", "GIRAFFE")), # Those outside of their regime (as determined by them, not other nodes)
    ("PHOTOMETRY", (load_node_data("data/photometry/GES_CoRoT_PhotTeff.fits", 1), "NONE_MARKED_AS_INVALID", "GIRAFFE")), # None INVALID
    ("ARCETRI", (load_node_data("data/arcetri/GES_CoRoT_Arcetri_iter1.fits", 1), "INVALID", "GIRAFFE")), # TECH flag of 1 or 2
])

assert len(set([each[2] for each in uves_results.values()])) == 1
assert len(set([each[2] for each in giraffe_results.values()])) == 1

bm_object, bm_teff, bm_u_teff, bm_logg, bm_u_logg, bm_z = np.loadtxt("data/benchmarks.txt",
    usecols=(2, 3, 4, 5, 6, 7), dtype=str, unpack=True)

# Convert data types
bm_teff, bm_u_teff, bm_logg, bm_u_logg, bm_z = map(lambda each: np.array(each, dtype=float),
    [bm_teff, bm_u_teff, bm_logg, bm_u_logg, bm_z])

# Create record array
benchmark_truths = np.core.records.fromarrays([bm_object, bm_teff, bm_u_teff, bm_logg, bm_u_logg, bm_z],
    names=["OBJECT", "TEFF", "u_TEFF", "LOGG", "u_LOGG", "FeH"], formats=["|S10"] + ["f8"] * 5)

initial_parameters = pyfits.open("data/initial_parameters.fits")[1].data

[validate(node, uves_results) for node in uves_results.keys()]
[validate(node, giraffe_results) for node in giraffe_results.keys()]
