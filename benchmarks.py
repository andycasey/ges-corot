
import logging

import numpy as np
import matplotlib.pyplot as plt

from load_data import benchmark_truths, all_node_results


def plot_benchmark_parameters():

    ax_limits = {
        "TEFF": (3500, 7000),
        "LOGG": (0, 5),
        "FeH": (-3, 0.5)
    }

    for node_name, (node_data, ignore_flags, setup) in all_node_results.iteritems():

        # Identify benchmarks in each node, and sort them properly.
        parameters = ("TEFF", "LOGG", "FeH")
        benchmark_indices = []
        benchmark_truth_indices = []
        benchmark_measurements = {}
        for parameter in parameters:
            benchmark_measurements[parameter] = []
        for i, benchmark_name in enumerate(benchmark_truths["OBJECT"]):
            index = np.where((node_data["TARGET"] == "GE_SD_BM") * (node_data["SETUP"] == setup) \
                * (node_data["OBJECT"] == benchmark_name))[0]
            if len(index) == 0:
                logging.warn("Benchmark {0} not found in {1} results.".format(benchmark_name, node_name))
                continue
            index = index[0]
            benchmark_indices.append(index)
            benchmark_truth_indices.append(i)
            for parameter in parameters:
                benchmark_measurements[parameter].append(node_data[parameter][index])

        # Did we miss any benchmarks?
        measured_benchmarks = (node_data["TARGET"] == "GE_SD_BM") * (node_data["SETUP"] == setup)
        difference = set(node_data["OBJECT"][measured_benchmarks]) - set(benchmark_truths["OBJECT"])
        if sum(measured_benchmarks) > len(benchmark_indices):
            logging.warn("There are {0} benchmarks from {1} node that we don't have truths for: {2}".format(
                len(difference), node_name, ", ".join(difference)))

        # Check for missing measurements.
        for parameter in parameters:
            non_finite_measurements = sum(~np.isfinite(node_data[parameter][benchmark_indices]))
            if non_finite_measurements > 0:
                logging.warn("Missing {0}/{1} {2} benchmark measurements from {3} node".format(
                    non_finite_measurements, len(benchmark_truths["OBJECT"]), parameter, node_name))

        # Plot the measured values compared to the benchmark truth values.
        fig, axes = plt.subplots(len(parameters), 1)
        for i, (ax, parameter) in enumerate(zip(axes, parameters)):

            ax.scatter(benchmark_truths[parameter][benchmark_truth_indices],
                benchmark_measurements[parameter])

            current_limits = [
                min([ax.get_xlim()[0], ax.get_ylim()[0]]),
                max([ax.get_xlim()[1], ax.get_ylim()[1]])
            ]
            limits = ax_limits.get(parameter, current_limits)
            ax.plot(limits, limits, c="k", ls=":", zorder=-1)

            ax.set_xlim(*limits)
            ax.set_ylim(*limits)
            ax.set_xlabel("{0} (True)".format(parameter))
            ax.set_ylabel("{0} (Measured)".format(parameter))

        axes[0].set_title("{0} ({1})".format(node_name, setup))
        fig.savefig("plots/benchmarks-{0}.pdf".format(node_name))
        plt.close(fig)


if __name__ == "__main__":
    plot_benchmark_parameters()

