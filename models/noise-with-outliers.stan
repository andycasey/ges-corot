
data {
    int<lower=1> N_nodes;
    int<lower=2> N_benchmarks;
    vector[N_benchmarks] non_spec_teff_measured;
    vector[N_benchmarks] non_spec_teff_sigma;
    matrix[N_nodes,N_benchmarks] node_teff_measured;
    matrix[N_nodes,N_benchmarks] additive_variance;
}

parameters {
    // Intrinsic uncertainty in the data
    real<lower=0> var_intrinsic;

    // Uncertainty due to each node
    vector<lower=0>[N_nodes] var_node;

    // God's word.
    vector[N_benchmarks] teff_truths;

    // Outliers. Can't live with 'em.
    real<lower=0,upper=1> outlier_fraction;
    real<lower=min(non_spec_teff_measured),upper=max(non_spec_teff_measured)> outlier_mu;
    real<lower=0,upper=10000000> outlier_variance;
}

model {

    // Initialise matrices
    matrix[N_nodes,N_nodes] covariance;
    matrix[N_nodes,N_nodes] outlier_covariance;

    // Benchmark truths
    teff_truths ~ normal(non_spec_teff_measured, non_spec_teff_sigma);

    for (i in 1:N_benchmarks) {
        covariance <- rep_matrix(var_intrinsic, N_nodes, N_nodes) + diag_matrix(var_node) + diag_matrix(col(additive_variance, i));
        outlier_covariance <- diag_matrix(rep_vector(var_intrinsic + outlier_variance, N_nodes)) + diag_matrix(var_node) + diag_matrix(col(additive_variance, i));

        // With outliers.
        increment_log_prob(log_sum_exp(
            log1m(outlier_fraction) + multi_normal_log(col(node_teff_measured, i), rep_vector(teff_truths[i], N_nodes), covariance),
             log(outlier_fraction)  + multi_normal_log(col(node_teff_measured, i), rep_vector(outlier_mu, N_nodes), outlier_covariance)
        ));
    }

}
