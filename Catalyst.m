[n_samples, n_features] = size(X);


minibatch_size = 50;
lambda = 1e-8;
x_old = zeros(n_features, 1);
y_old = zeros(n_features, 1);
max_outer_iter = 10;
max_inner_iter = 5;
sigma_sc = 1e-8;

X_t = X'; 

XX = X_t.^2;
weight = sum(XX, 1)';
Lipschitz_constant = mean(full(weight));

parameter.kappa = Lipschitz_constant/n_samples - sigma_sc;


optimal_objective_value = 0;
epsilon_suboptimality = 0;

phi_handle = @(linear_activation, response) norm(linear_activation - response, 2)^2/2;
dphi_handle = @(linear_activation, response) (linear_activation - response);
Threshold_Handle = @(w, step_size) wthresh(w, 's', lambda*step_size);

p_err_Catalyst_SVRG = zeros(max_outer_iter * max_inner_iter + 1, 1);
epoch_Catalyst_SVRG = zeros(max_outer_iter * max_inner_iter + 1, 1);
time_Catalyst_SVRG = zeros(max_outer_iter * max_inner_iter + 1, 1);
nnz_Catalyst_SVRG = zeros(max_outer_iter * max_inner_iter + 1, 1);

q = sigma_sc/(sigma_sc+parameter.kappa);
alpha_old = sqrt(q);
for outer_iter = 1:max_outer_iter
    [x_new, p_err_Catalyst_SVRG_in, epoch_Catalyst_SVRG_in, time_Catalyst_SVRG_in, nnz_Catalyst_SVRG_in] ...
        = Catalyst_SVRG(y_old, x_old, phi_handle, dphi_handle,...
        Threshold_Handle, X, X_t, y, optimal_objective_value, epsilon_suboptimality, max_inner_iter,...
        parameter.kappa, sigma_sc, lambda, Lipschitz_constant, minibatch_size);
    if outer_iter == 1
        begin_index = 1; end_index = max_inner_iter+1;
        p_err_Catalyst_SVRG(begin_index: end_index) = p_err_Catalyst_SVRG_in;
        epoch_Catalyst_SVRG(begin_index: end_index) = epoch_Catalyst_SVRG_in;
        time_Catalyst_SVRG(begin_index: end_index) = time_Catalyst_SVRG_in;
        nnz_Catalyst_SVRG(begin_index: end_index) = nnz_Catalyst_SVRG_in;
    else
        last_end_index = end_index;
        begin_index = end_index + 1; end_index = end_index + max_inner_iter;
        p_err_Catalyst_SVRG(begin_index: end_index) = p_err_Catalyst_SVRG_in(2:end);
        epoch_Catalyst_SVRG(begin_index: end_index) = epoch_Catalyst_SVRG_in(2:end) + epoch_Catalyst_SVRG(last_end_index);
        time_Catalyst_SVRG(begin_index: end_index) = time_Catalyst_SVRG_in(2:end) + time_Catalyst_SVRG(last_end_index);
        nnz_Catalyst_SVRG(begin_index: end_index) = nnz_Catalyst_SVRG_in(2:end);
    end
    for it = begin_index:end_index
        msg = sprintf('%d th epoch, primal obj %.4e, nnz %d, time %.4e', epoch_Catalyst_SVRG(it), p_err_Catalyst_SVRG(it), nnz_Catalyst_SVRG(it), time_Catalyst_SVRG(it)); disp(msg);
    end
    temp = alpha_old^2 - q;
    alpha_new = (-temp + sqrt(temp + 4*alpha_old^2))/2;
    beta_k = alpha_old*(1-alpha_old)/(alpha_old^2 + alpha_new);
    y_old = x_new + beta_k*(x_new - x_old);
    x_old = x_new;
end