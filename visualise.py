
""" Visualise the node data prior to homogenisation. """

import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from load_data import all_node_results
from itertools import combinations

import numpy as np


def node_benchmark_measurements(benchmarks, model, recommended_values=None, recommended_uncertainties=None,
    colours=("k", "g", "r"), extent=500, extent_offset=50):

    num_nodes, num_benchmarks = model.data["node_teff_measured"].shape
    fig, axes = plt.subplots(num_nodes, 1, figsize=(8, 14))
    fig.subplots_adjust(right=0.98, top=0.98, bottom=0.10, left=0.10)

    # Grey is uncertainty range given by non-spectroscopic methods
    # Green is uncertainty of the recommended measurement

    # Sort benchmarks by temperature
    benchmark_indices = np.argsort(benchmarks["TEFF"])
    
    # Sort by alphabetical node name
    for i, (ax, node_name) in enumerate(zip(axes, sorted(model.data["node_names"]))):

        node_index = model.data["node_names"].index(node_name)

        # Draw the grey (non-spectroscopic) uncertainty regions for each plot.
        patches = []
        for j, benchmark in enumerate(benchmarks[benchmark_indices], start=1):
            patches.append(Polygon([
                [ j - 0.225, - benchmark["u_TEFF"]],
                [ j - 0.225, + benchmark["u_TEFF"]],
                [ j + 0.225, + benchmark["u_TEFF"]],
                [ j + 0.225, - benchmark["u_TEFF"]]
            ], True))
        ax.add_collection(PatchCollection(patches, alpha=0.4, facecolor="#666666", edgecolors=None,
            linewidths=0, zorder=100))

        # Plot the point uncertainty as measured by the node.
        node_measurement = model.data["node_teff_measured"][node_index, benchmark_indices]
        node_uncertainty = model.data["node_teff_uncertainty"][node_index, benchmark_indices]
        node_difference = node_measurement - benchmarks[benchmark_indices]["TEFF"]
        
        bad_measurements = (0 >= node_measurement) * np.isfinite(node_measurement)
        node_difference[bad_measurements] = np.nan
        node_uncertainty[bad_measurements] = np.nan

        ax.plot([0, num_benchmarks + 1], [0, 0], ":", c="#666666", zorder=-100)
        #within extent
        within_indices = np.where(np.abs(node_difference) < extent)[0]
        ax.errorbar(1 + within_indices, node_difference[within_indices], yerr=node_uncertainty[within_indices],
            fmt=None, c=colours[0], elinewidth=1, capsize=2, ecolor=colours[0], zorder=1000)
        ax.scatter(1 + within_indices, node_difference[within_indices], facecolor=colours[0], zorder=10000)

        # Any outside of extent?
        if np.any(node_difference > extent):
            outside_indices = np.where(node_difference > extent)[0]
            ax.scatter(1 + outside_indices, [extent - extent_offset] * len(outside_indices), facecolor=colours[2],
                marker="^", edgecolor=colours[2], zorder=100000)

        if np.any(-extent > node_difference):
            outside_indices = np.where(-extent > node_difference)[0]
            ax.scatter(1 + outside_indices, [-extent + extent_offset] * len(outside_indices), facecolor=colours[2],
                marker="v", edgecolor=colours[2], zorder=100000)
        
        # If we have recommended values, plot their value and region.
        if recommended_uncertainties is not None and recommended_values is not None:
            patches = []
            for j, (y, y_uncertainty) in enumerate(zip(recommended_values[benchmark_indices] - benchmarks["TEFF"][benchmark_indices], \
            recommended_uncertainties[benchmark_indices]), start=1):
                patches.append(Polygon([
                    [ j - 0.225, y - y_uncertainty],
                    [ j - 0.225, y + y_uncertainty],
                    [ j + 0.225, y + y_uncertainty],
                    [ j + 0.225, y - y_uncertainty]
                ], True, facecolor=colours[1]))
                ax.plot([j - 0.225 +0.05, j + 0.225 - 0.05], [y, y], lw=2, c=colours[1])

            ax.add_collection(PatchCollection(patches, alpha=0.4, facecolor=colours[1], edgecolors=colours[1], linewidths=0, zorder=100))
            
        # Write the node name.
        print(node_name)
        # Prettify the axes
        ax.set_xlim(0.5, num_benchmarks + 0.5)
        ax.set_ylim(-extent, +extent)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.yaxis.set_major_locator(MaxNLocator(3))
        node_label = node_name.replace("-", " ")
        ax.set_ylabel(node_label)# + "\n\n$\Delta{}T_{\\rm eff}\,({\\rm K})$")
        #ax.yaxis.set_label_coords(-0.025, 0.5)

        [l.set_rotation(45) for l in ax.get_yticklabels()]

        ax.spines["left"]._linewidth = 0.5
        ax.spines["bottom"]._linewidth = 0.0
        ax.spines["top"]._linewidth = 0.0
        ax.spines["right"]._linewidth = 0.0

        opposite_ax = ax.twinx()
        opposite_ax.set_yticks([])
        #opposite_ax.set_ylabel(node_name, rotation=0)
        #ax.set_title(node_name, loc='right')

    ax.set_xticks(1 + np.arange(num_benchmarks))    
    ax.spines["bottom"]._linewidth = 0.5
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_ticklabels([label.replace("_", " ") for label in benchmarks["OBJECT"][benchmark_indices]],
        rotation=90)

    return fig


def boxplots(benchmark_parameters, model, recommended_values=None, recommended_uncertainties=None, sort=True,
    summarise=False, colours=("k", "g")):
    """ Do box plots """

    dimension = "TEFF"
    num_nodes, num_benchmarks = model.data["node_teff_measured"].shape

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.20, right=0.95, top=0.95)
    
    measurements = model.data["node_teff_measured"]
    measurements[0 >= measurements] = np.nan
    deltas = measurements - benchmark_parameters[dimension.upper()]

    # Sort?
    sort_indices = np.arange(len(deltas)) if not sort else np.argsort(benchmark_parameters[dimension.upper()])
    deltas = deltas[:, sort_indices]

    ax.plot([0, num_benchmarks + 1], [0, 0], ":", c="#666666", zorder=-10)
    width = 0.45
    bp = ax.boxplot([[v for v in row if np.isfinite(v)] for row in deltas.T], widths=width,
        patch_artist=True)

    assert len(bp["boxes"]) == num_benchmarks
    
    ylims = ax.get_ylim()
    # Get 5% in y-direction
    text_y_percent = 3
    text_y_position = (text_y_percent/100.) * np.ptp(ylims) + ylims[0]
    for j in xrange(num_benchmarks):
        num_finite = np.sum(np.isfinite(deltas[:, j]))
        ax.text(j + 1, text_y_position, num_finite, size=10,
            horizontalalignment="center")

    text_y_position = (2.5*text_y_percent/100.) * np.ptp(ylims) + ylims[0]
    ax.text(num_benchmarks/2., text_y_position, "Number of node measurements", size=10,
        horizontalalignment="center", fontweight="bold")
    
    # Set y-lims back to where they should be
    ax.set_ylim(ylims)

    # Hide spines and tick positions
    ax.xaxis.set_ticks_position("none")
    
    # Label axes
    ax.set_ylabel("$\Delta{}T_{\\rm eff}\,({\\rm K})$")
    ax.xaxis.set_ticklabels([label.replace("_", " ") for label in benchmark_parameters["OBJECT"][sort_indices]],
        rotation=90)
    
    # Set colours
    plt.setp(bp["medians"], color=colours[0], linewidth=2)
    plt.setp(bp["fliers"], color=colours[0])
    plt.setp(bp["caps"], visible=False)
    plt.setp(bp["whiskers"], color=colours[0], linestyle="solid", linewidth=0.5)
    plt.setp(bp["boxes"], color=colours[0], linewidth=2, facecolor="w")

    ax.spines["left"]._linewidth = 0.5
    ax.spines["bottom"]._linewidth = 0.5
    ax.spines["top"]._linewidth = 0.5
    ax.spines["right"]._linewidth = 0.5
    
    # Draw recommended values if they exist
    if recommended_uncertainties is not None and recommended_values is not None:
        patches = []
        for x_center, y, y_uncertainty in zip(np.arange(1, num_benchmarks + 1), \
        recommended_values[sort_indices] - benchmark_parameters[dimension][sort_indices], recommended_uncertainties[sort_indices]):
            points = np.array([
                [ x_center - 0.225, y - y_uncertainty],
                [ x_center - 0.225, y + y_uncertainty],
                [ x_center + 0.225, y + y_uncertainty],
                [ x_center + 0.225, y - y_uncertainty]
            ])
            patches.append(Polygon(points, True, facecolor=colours[1]))
        ax.add_collection(PatchCollection(patches, alpha=0.4, facecolor=colours[1], edgecolors=None, linewidths=0, zorder=100))
        
    if recommended_values is not None:
        for x_center, y in zip(np.arange(1, num_benchmarks + 1), recommended_values[sort_indices] - benchmark_parameters[dimension][sort_indices]):
            ax.plot([x_center - width/2., width/2. + x_center], [y, y], lw=2, c=colours[1])
    
    # Put temperatures along the top x-axis if we are sorted.
    if sort:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        pos = [4000, 5000, 6000]
        ax2.set_xticks(np.arange(1, num_benchmarks + 1)[benchmark_parameters[dimension][sort_indices].searchsorted(pos)] - 0.5)
        ax2.set_xticklabels(map(str, pos))
        ax2.set_xlabel("$T_{\\rm eff}\,({\\rm K})$")
        fig.subplots_adjust(top=0.90)

    [l.set_rotation(45) for l in ax.get_yticklabels()]
    ax.set_ylim(ax.get_ylim()[0] + 0.001, ax.get_ylim()[1] - 0.001)
    return fig


def inferred_node_uncertainties(model, node_names=None):
    """
    Plot the inferred uncertainties due to each node.
    """
    if node_names is None:
        node_names = model.data["node_names"]

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

    return fig


def node_all_measurements(all_node_results, parameter="TEFF", extent=(3000, 7000)):
    """
    Create a corner plot showing the differences between each node.
    """

    # How many nodes to plot?
    K = len(nodes) - 1
    assert K > 1, "Need more than one node to compare against."

    factor = 2.0           # size of one side of one panel
    lbdim = 0.5 * factor   # size of left/bottom margin
    trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.15         # w/hspace size
    plotdim = factor * K + factor * (K - 1.) * whspace
    dim = lbdim + plotdim + trdim

    fig, axes = plt.subplots(K, K, figsize=(dim, dim))
    
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
        wspace=whspace, hspace=whspace)
    
    for i, node_y in enumerate(all_node_results.keys()):

        for j, node_x in enumerate(all_node_results.keys()):
            if j == K: break
            elif j > i:
                try:
                    ax = axes[i, j]
                except IndexError:
                    continue
                ax.set_visible(False)
                ax.set_frame_on(False)
                continue
            if 0 > i-1: continue
            #print(i, j, node_x, node_y)
            ax = axes[i-1, j]
            
            indices = all_node_results[node_x][0]["SETUP"] == setup
            x_data = all_node_results[node_x][0][parameter][indices]
            y_data = all_node_results[node_y][0][parameter][indices]
            x_err = all_node_results[node_x][0]["e_"+parameter][indices]
            y_err = all_node_results[node_y][0]["e_"+parameter][indices]

            ax.plot(extent, extent, "k:", zorder=-100)
            ax.errorbar(x_data, y_data, xerr=x_err, yerr=y_err, c="k", aa=True, fmt='o', mec='k',
                mfc="w", ms=6, zorder=100)
            
            ax.set_xlim(extent)
            ax.set_ylim(extent)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))

            if i != K:
                ax.set_xticklabels([])
            else:
                [l.set_rotation(45) for l in ax.get_xticklabels()]
                ax.set_xlabel(node_x)
                ax.xaxis.set_label_coords(0.5, -0.3)
                
            if j > 0:
                ax.set_yticklabels([])
            else:
                [l.set_rotation(45) for l in ax.get_yticklabels()]
                ax.set_ylabel(node_y)
                ax.yaxis.set_label_coords(-0.3, 0.5)

    return fig


def plot_uncertainty_distributions(all_node_results, setup, parameter="TEFF"):

    # Get only nodes from this setup.
    nodes = []
    for node_name, (node_data, node_flags, node_setup) in all_node_results.iteritems():
        if node_setup == setup:
            nodes.append(node_name)

    K = len(nodes)
    factor = 2.0           # size of one side of one panel
    lbdim = 0.5 * factor   # size of left/bottom margin
    trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.15         # w/hspace size
    height = factor * K + factor * (K - 1.) * whspace
    width = factor * K 

    dimy = lbdim + height + trdim
    dimx = lbdim + width + trdim 

    fig, axes = plt.subplots(K, 1, figsize=(dimx, dimy))
    
    lb = lbdim / dimx
    tr = (lbdim + height) / dimy
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
        wspace=whspace, hspace=whspace)

    bins = np.arange(0, 325, 25)
    for node, ax in zip(nodes, axes):

        indices = all_node_results[node][0]["SETUP"] == setup
        data = all_node_results[node][0]["e_" + parameter][indices]

        finite_data = data[np.isfinite(data) * (data > 0)]
        ax.hist(finite_data, bins=bins)
        ax.set_xlim(bins[0], bins[-1])

        ax.set_ylabel(node)
        if node != nodes[-1]:
            ax.set_xticklabels([])

        else:
            [l.set_rotation(45) for l in ax.get_xticklabels()]
            ax.xaxis.set_label_coords(0.5, -0.3)

        [l.set_rotation(45) for l in ax.get_yticklabels()]
        #ax.yaxis.set_label_coords(-0.3, 0.5)

    y_max = max([ax.get_ylim()[1] for ax in axes])
    [ax.set_ylim(0, y_max) for ax in axes]

    ax.set_xlabel("{0} UNCERTAINTY".format(parameter))
    return fig

"""
if __name__ == "__main__":

    fig = plot_uncertainty_distributions(all_node_results, "UVES")
    fig.savefig("plots/uves-uncertainty-distribution.pdf")
    fig = plot_uncertainty_distributions(all_node_results, "GIRAFFE")
    fig.savefig("plots/giraffe-uncertainty-distribution.pdf")

    uves_figure = visualise_by_setup(all_node_results, "UVES")
    uves_figure.savefig("plots/uves-comparison.pdf")
    giraffe_figure = visualise_by_setup(all_node_results, "GIRAFFE")
    giraffe_figure.savefig("plots/giraffe-comparison.pdf")
"""
