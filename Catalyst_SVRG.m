function [x, p_err, epoch, time_table, nnz_table] ...
        = Catalyst_SVRG(x_0, x_init, phi_handle, dphi_handle,...
        Threshold_Handle, data_matrix, data_matrix_t, response_vector, optimal_objective_value, epsilon_suboptimality, max_iteration,...
        kappa, sigma_sc, lambda, Lipschitz_constant, minibatch_size)
    kappa_negate_x_0 = -x_0 * kappa;
    [n_samples, ~] = size(data_matrix);
    inner_loop_count = ceil(n_samples/minibatch_size);     
    x_tilde = x_init;
    x = x_tilde;
    epoch_sum = 0;
    epoch = zeros(max_iteration+1, 1);
    p_err = zeros(max_iteration+1, 1);
    time_sum = 0;
    time_table = zeros(max_iteration+1, 1);
    nnz_table = zeros(max_iteration+1, 1);
                                                                            time_b = tic;
    data_matrix_x_tilde = data_matrix_t'*x_tilde;
    p_err(1) = phi_handle(data_matrix_x_tilde, response_vector)/n_samples + lambda*norm(x_tilde, 1) + norm(x_tilde, 2)^2*sigma_sc/2;
    dphi_value = dphi_handle(data_matrix_x_tilde, response_vector);
    full_gradient = data_matrix'*dphi_value/n_samples;
                                                                            time_e = toc(time_b);
                                                                            time_sum = time_sum + time_e;
    epoch_sum = epoch_sum + 1;
    nnz_table(1) = nnz(x_tilde);
%     msg = sprintf('%d th epoch, primal obj %.4e, nnz %d, time %.4e', epoch(1), p_err(1), nnz_table(1), time_table(1)); disp(msg);
    %----------------------------------------------------------------------
    for it = 1:max_iteration
                                                                            time_b = tic;
        step_size = 1/Lipschitz_constant;
        for it_in = 1:inner_loop_count
            MB_start = randi(n_samples-minibatch_size+1, 1);
            MBIndex = MB_start:(MB_start+minibatch_size-1);
            
            minibatch_data_matrix_x = data_matrix_t(:, MBIndex)'*x;
            dphi_value_minibatch = dphi_handle(minibatch_data_matrix_x, response_vector(MBIndex));
            gradient_delta = data_matrix_t(:, MBIndex)*(dphi_value_minibatch - dphi_value(MBIndex))/minibatch_size;
            gradient_in = gradient_delta + full_gradient + kappa*x + kappa_negate_x_0;
            gradient_in = gradient_in + sigma_sc*x;
            
            if lambda ~=0
                x = Threshold_Handle(x - step_size*gradient_in, step_size);
            else
                x = x - step_size*gradient_in;
            end
        end
        x_tilde = x;
                                                                            time_e = toc(time_b);
                                                                            time_sum = time_sum + time_e;
        epoch_sum = epoch_sum + inner_loop_count*minibatch_size/n_samples;
        epoch(it+1) = epoch_sum;
        
        time_table(it+1) = time_sum;
        nnz_table(it+1) = nnz(x_tilde);
        
                                                                            time_b = tic;
        data_matrix_x_tilde = data_matrix_t'*x;
        p_err(it+1) = phi_handle(data_matrix_x_tilde, response_vector)/n_samples + lambda*norm(x, 1) + norm(x, 2)^2*sigma_sc/2;
        dphi_value = dphi_handle(data_matrix_x_tilde, response_vector);
        full_gradient = data_matrix'*dphi_value/n_samples;
                                                                            time_e = toc(time_b);
                                                                            time_sum = time_sum + time_e;
        epoch_sum = epoch_sum + 1;
%         msg = sprintf('%d th epoch, primal obj %.4e, nnz %d, time %.4e', epoch(it+1), p_err(it+1), nnz_table(it+1), time_table(it+1)); disp(msg);
        if p_err(it+1) - optimal_objective_value < epsilon_suboptimality
            break;
        end
    end
    p_err = p_err(1:it+1);
    epoch = epoch(1:it+1);
    time_table = time_table(1:it+1);
    nnz_table = nnz_table(1:it+1);
end