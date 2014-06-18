
import logging
import numpy as np
import pyfits

__all__ = ["all_node_results", "benchmarks", "initial_parameters", "validate_all"]

all_node_results = {
    "LUMBA-1": (pyfits.open("node-data/lumba/GES_CoRoT_25May_LUMBA.fits/GES_CoRoT_25May_LUMBA.fits")[1].data, None, "GIRAFFE"),
    "LUMBA-2": (pyfits.open("node-data/lumba/GES_CoRoT_25May_LUMBA.fits/GES_CoRoT_25May_LUMBA.fits")[2].data, None, "GIRAFFE"),
    "BOLOGNA": (pyfits.open("node-data/bologna/Bologna_GES_CoRoT/Bologna_GES_CoRoT/GES_CoRoT_25May2014_Bologna.fits")[1].data, None, "UVES"),
    "VILNIUS": (pyfits.open("node-data/vilnius/GES_Corot_iteration1_Vilnius.v11-corrected.fits")[1].data, None, "UVES"),
    "EPINARBO-UVES": (pyfits.open("node-data/epinarbo/Results_CoRoT_Epinarbo_18June2014/GES_CoRoT_Epinarbo_18June2014.fits")[1].data, None, "UVES"),
    "EPINARBO-GIRAFFE": (pyfits.open("node-data/epinarbo/Results_CoRoT_Epinarbo_18June2014/GES_CoRoT_Epinarbo_18June2014.fits")[1].data, None, "GIRAFFE"),
}

bm_object, bm_teff, bm_u_teff, bm_logg, bm_u_logg, bm_z = np.loadtxt("benchmarks.txt",
    usecols=(2, 3, 4, 5, 6, 7), dtype=str, unpack=True)

# Convert data types
bm_teff, bm_u_teff, bm_logg, bm_u_logg, bm_z = map(lambda each: np.array(each, dtype=float),
    [bm_teff, bm_u_teff, bm_logg, bm_u_logg, bm_z])

# Create record array
benchmark_truths = np.core.records.fromarrays([bm_object, bm_teff, bm_u_teff, bm_logg, bm_u_logg, bm_z],
    names=["OBJECT", "TEFF", "u_TEFF", "LOGG", "u_LOGG", "FeH"],
    formats=["|S10"] + ["f8"] * 5)

initial_parameters = pyfits.open("node-data/initial_parameters.fits")[1].data

def validate(node):
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

def validate_all():
    return map(validate, all_node_results.keys())