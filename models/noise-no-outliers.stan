
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
real<lower=0,upper=max(non_spec_teff_measured) - min(non_spec_teff_measured)> s_intrinsic;

// Uncertainty due to each node
vector<lower=0>[N_nodes] var_teff_node;

// God's word.
vector[N_benchmarks] teff_truths;
}

model {

    // Initialise matrices
    matrix[N_nodes,N_nodes] covariance;

    // Benchmark truths
    teff_truths ~ normal(non_spec_teff_measured, non_spec_teff_sigma);

    for (i in 1:N_benchmarks) {
        covariance <- diag_matrix(var_teff_node) + rep_matrix(pow(s_intrinsic, 2), N_nodes, N_nodes) + diag_matrix(col(additive_variance, i));
        increment_log_prob(multi_normal_log(col(node_teff_measured, i), rep_vector(teff_truths[i], N_nodes), covariance));
    }

}
