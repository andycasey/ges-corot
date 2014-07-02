
import corot
import visualise
from load_data import benchmark_truths, uves_results

# Run name.
run_name = "uves-{0}".format(md5(os.path.basename(ctime()).encode("utf-8")).hexdigest())

# Draw all the node measurements first
fig_node_all_measurements = visualise.node_all_measurements(uves_results)

# Calibrate the model
model = corot.calibrate_noise_model(benchmark_truths, uves_results, run_name=run_name)

# Draw the node uncertainties
fig_node_uncertainties = visualise.inferred_node_uncertainties(model)

# Do a cross-validation of all the benchmark stars
cv_measurements, cv_uncertainties = corot.cross_validate(benchmark_truths, uves_results)

# Do some post-processing of the cross-validation results
fig_boxplots = visualise.boxplots(benchmark_truths, model, cv_measurements, cv_uncertainties)
fig_node_benchmark_measurements = visualise.node_benchmark_measurements(benchmark_truths, model,
    cv_measurements, cv_uncertainties)

# Homogenise all the data.
homogenised_results = corot.homogenise_all(uves_results, model)

# Visualise the homogenised results in context of the initial node results as a corner plot.

# Plot the homogenised uncertainties against S/N, magnitude


# Create a GES/CoRoT-standardised FITS table with the homogenised results.
