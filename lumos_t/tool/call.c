#include "call.h"

int call(char *interface, void **params, void **ret)
{
    if (0 == strcmp(interface, "add_bias")){
        call_add_bias(params, ret);
    } else if (0 == strcmp(interface, "fill_cpu")){
        call_fill_cpu(params, ret);
    } else if (0 == strcmp(interface, "multy_cpu")){
        call_multy_cpu(params, ret);
    } else if (0 == strcmp(interface, "add_cpu")){
        call_add_cpu(params, ret);
    } else if (0 == strcmp(interface, "min_cpu")){
        call_min_cpu(params, ret);
    } else if (0 == strcmp(interface, "max_cpu")){
        call_max_cpu(params, ret);
    } else if (0 == strcmp(interface, "sum_cpu")){
        call_sum_cpu(params, ret);
    } else if (0 == strcmp(interface, "mean_cpu")){
        call_mean_cpu(params, ret);
    } else if (0 == strcmp(interface, "matrix_add_cpu")){
        call_matrix_add_cpu(params, ret);
    } else if (0 == strcmp(interface, "matrix_subtract_cpu")){
        call_matrix_subtract_cpu(params, ret);
    } else if (0 == strcmp(interface, "matrix_multiply_cpu")){
        call_matrix_multiply_cpu(params, ret);
    } else if (0 == strcmp(interface, "matrix_divide_cpu")){
        call_matrix_divide_cpu(params, ret);
    } else if (0 == strcmp(interface, "saxpy_cpu")){
        call_saxpy_cpu(params, ret);
    } else if (0 == strcmp(interface, "sum_channel_cpu")){
        call_sum_channel_cpu(params, ret);
    } else if (0 == strcmp(interface, "one_hot_encoding")){
        call_one_hot_encoding(params, ret);
    } else if (0 == strcmp(interface, "gemm")){
        call_gemm(params, ret);
    } else if (0 == strcmp(interface, "gemm_nn")){
        call_gemm_nn(params, ret);
    } else if (0 == strcmp(interface, "gemm_nt")){
        call_gemm_nt(params, ret);
    } else if (0 == strcmp(interface, "gemm_tn")){
        call_gemm_tn(params, ret);
    } else if (0 == strcmp(interface, "gemm_tt")){
        call_gemm_tt(params, ret);
    } else if (0 == strcmp(interface, "im2col")){
        call_im2col(params, ret);
    } else if (0 == strcmp(interface, "col2im")){
        call_col2im(params, ret);
    } else if (0 == strcmp(interface, "avgpool")){
        call_avgpool(params, ret);
    } else if (0 == strcmp(interface, "maxpool")){
        call_maxpool(params, ret);
    } else if (0 == strcmp(interface, "avgpool_gradient")){
        call_avgpool_gradient(params, ret);
    } else if (0 == strcmp(interface, "maxpool_gradient")){
        call_maxpool_gradient(params, ret);
    } else if (0 == strcmp(interface, "forward_avgpool_layer")){
        call_forward_avgpool_layer(params, ret);
    } else if (0 == strcmp(interface, "backward_avgpool_layer")){
        call_backward_avgpool_layer(params, ret);
    } else if (0 == strcmp(interface, "forward_connect_layer")){
        call_forward_connect_layer(params, ret);
    } else if (0 == strcmp(interface, "backward_connect_layer")){
        call_backward_connect_layer(params, ret);
    } else if (0 == strcmp(interface, "forward_convolutional_layer")){
        call_forward_convolutional_layer(params, ret);
    } else if (0 == strcmp(interface, "backward_convolutional_layer")){
        call_backward_convolutional_layer(params, ret);
    } else if (0 == strcmp(interface, "forward_im2col_layer")){
        call_forward_im2col_layer(params, ret);
    } else if (0 == strcmp(interface, "backward_im2col_layer")){
        call_backward_im2col_layer(params, ret);
    } else if (0 == strcmp(interface, "forward_maxpool_layer")){
        call_forward_maxpool_layer(params, ret);
    } else if (0 == strcmp(interface, "backward_maxpool_layer")){
        call_backward_maxpool_layer(params, ret);
    } else if (0 == strcmp(interface, "forward_mse_layer")){
        call_forward_mse_layer(params, ret);
    } else if (0 == strcmp(interface, "dropout_rand")){
        call_dropout_rand(params, ret);
    } else if (0 == strcmp(interface, "layer_delta")){
        call_layer_delta(params, ret);
    } else if (0 == strcmp(interface, "loss")){
        call_loss(params, ret);
    } else if (0 == strcmp(interface, "maxpool_index")){
        call_maxpool_index(params, ret);
    } else if (0 == strcmp(interface, "mean")){
        call_mean(params, ret);
    } else if (0 == strcmp(interface, "output")){
        call_output(params, ret);
    } else if (0 == strcmp(interface, "roll_mean")){
        call_roll_mean(params, ret);
    } else if (0 == strcmp(interface, "roll_variance")){
        call_roll_variance(params, ret);
    } else if (0 == strcmp(interface, "truth")){
        call_truth(params, ret);
    } else if (0 == strcmp(interface, "update_weights")){
        call_update_weights(params, ret);
    } else if (0 == strcmp(interface, "variance")){
        call_variance(params, ret);
    } else if (0 == strcmp(interface, "weights")){
        call_weights(params, ret);
    } else if (0 == strcmp(interface, "x_norm")){
        call_x_norm(params, ret);
    } else {
        fprintf(stderr, "  interface: %s is not in the testlist\n", interface);
        return 0;
    }
    return 1;
}

int call_cu(char *interface, void **params, void **ret)
{
    if (0 == strcmp(interface, "add_bias")){
        call_add_bias_gpu(params, ret);
    } else if (0 == strcmp(interface, "add_cpu")){
        call_add_gpu(params, ret);
    } else if (0 == strcmp(interface, "fill_cpu")){
        call_fill_gpu(params, ret);
    } else if (0 == strcmp(interface, "min_cpu")){
        call_min_gpu(params, ret);
    } else if (0 == strcmp(interface, "max_cpu")){
        call_max_gpu(params, ret);
    } else if (0 == strcmp(interface, "sum_cpu")){
        call_sum_gpu(params, ret);
    } else if (0 == strcmp(interface, "mean_cpu")){
        call_mean_gpu(params, ret);
    } else if (0 == strcmp(interface, "matrix_add_cpu")){
        call_matrix_add_gpu(params, ret);
    } else if (0 == strcmp(interface, "matrix_divide_cpu")){
        call_matrix_divide_gpu(params, ret);
    } else if (0 == strcmp(interface, "matrix_multiply_cpu")){
        call_matrix_multiply_gpu(params, ret);
    } else if (0 == strcmp(interface, "matrix_subtract_cpu")){
        call_matrix_subtract_gpu(params, ret);
    } else if (0 == strcmp(interface, "multy_cpu")){
        call_multy_gpu(params, ret);
    } else if (0 == strcmp(interface, "saxpy_cpu")){
        call_saxpy_gpu(params, ret);
    } else if (0 == strcmp(interface, "sum_channel_cpu")){
        call_sum_channel_gpu(params, ret);
    }else if (0 == strcmp(interface, "gemm")){
        call_gemm_gpu(params, ret);
    } else if (0 == strcmp(interface, "gemm_nn")){
        call_gemm_nn_gpu(params, ret);
    } else if (0 == strcmp(interface, "gemm_nt")){
        call_gemm_nt_gpu(params, ret);
    } else if (0 == strcmp(interface, "gemm_tn")){
        call_gemm_tn_gpu(params, ret);
    } else if (0 == strcmp(interface, "gemm_tt")){
        call_gemm_tt_gpu(params, ret);
    } else if (0 == strcmp(interface, "im2col")){
        call_im2col_gpu(params, ret);
    } else if (0 == strcmp(interface, "col2im")){
        call_col2im_gpu(params, ret);
    } else if (0 == strcmp(interface, "avgpool")){
        call_avgpool_gpu(params, ret);
    } else if (0 == strcmp(interface, "maxpool")){
        call_maxpool_gpu(params, ret);
    } else if (0 == strcmp(interface, "avgpool_gradient")){
        call_avgpool_gradient_gpu(params, ret);
    } else if (0 == strcmp(interface, "maxpool_gradient")){
        call_maxpool_gradient_gpu(params, ret);
    } else if (0 == strcmp(interface, "forward_avgpool_layer")){
        call_forward_avgpool_layer_gpu(params, ret);
    } else if (0 == strcmp(interface, "backward_avgpool_layer")){
        call_backward_avgpool_layer_gpu(params, ret);
    } else if (0 == strcmp(interface, "forward_connect_layer")){
        call_forward_connect_layer_gpu(params, ret);
    } else if (0 == strcmp(interface, "backward_connect_layer")){
        call_backward_connect_layer_gpu(params, ret);
    } else if (0 == strcmp(interface, "forward_convolutional_layer")){
        call_forward_convolutional_layer_gpu(params, ret);
    } else if (0 == strcmp(interface, "backward_convolutional_layer")){
        call_backward_convolutional_layer_gpu(params, ret);
    } else if (0 == strcmp(interface, "forward_im2col_layer")){
        call_forward_im2col_layer_gpu(params, ret);
    } else if (0 == strcmp(interface, "backward_im2col_layer")){
        call_backward_im2col_layer_gpu(params, ret);
    } else if (0 == strcmp(interface, "forward_maxpool_layer")){
        call_forward_maxpool_layer_gpu(params, ret);
    } else if (0 == strcmp(interface, "backward_maxpool_layer")){
        call_backward_maxpool_layer_gpu(params, ret);
    } else if (0 == strcmp(interface, "dropout_rand")){
        call_dropout_rand_gpu(params, ret);
    } else if (0 == strcmp(interface, "layer_delta")){
        call_layer_delta_gpu(params, ret);
    } else if (0 == strcmp(interface, "loss")){
        call_loss_gpu(params, ret);
    } else if (0 == strcmp(interface, "maxpool_index")){
        call_maxpool_index_gpu(params, ret);
    } else if (0 == strcmp(interface, "mean")){
        call_normalize_mean_gpu(params, ret);
    } else if (0 == strcmp(interface, "output")){
        call_output_gpu(params, ret);
    } else if (0 == strcmp(interface, "roll_mean")){
        call_roll_mean_gpu(params, ret);
    } else if (0 == strcmp(interface, "roll_variance")){
        call_roll_variance_gpu(params, ret);
    } else if (0 == strcmp(interface, "truth")){
        call_truth_gpu(params, ret);
    } else if (0 == strcmp(interface, "update_weights")){
        call_update_weights_gpu(params, ret);
    } else if (0 == strcmp(interface, "variance")){
        call_variance_gpu(params, ret);
    } else if (0 == strcmp(interface, "weights")){
        call_weights_gpu(params, ret);
    } else if (0 == strcmp(interface, "x_norm")){
        call_x_norm_gpu(params, ret);
    } else {
        fprintf(stderr, "  interface: %s is not in the testlist\n", interface);
        return 0;
    }
    return 1;
}

char *interface_to_benchmark(char *interface)
{
    if (0 == strcmp(interface, "add_bias")){
        return "./lumos_t/benchmark/core/ops/bias/add_bias.json";
    } else if (0 == strcmp(interface, "fill_cpu")){
        return "./lumos_t/benchmark/core/ops/cpu/fill_cpu.json";
    } else if (0 == strcmp(interface, "multy_cpu")){
        return "./lumos_t/benchmark/core/ops/cpu/multy_cpu.json";
    } else if (0 == strcmp(interface, "add_cpu")){
        return "./lumos_t/benchmark/core/ops/cpu/add_cpu.json";
    } else if (0 == strcmp(interface, "min_cpu")){
        return "./lumos_t/benchmark/core/ops/cpu/min_cpu.json";
    } else if (0 == strcmp(interface, "max_cpu")){
        return "./lumos_t/benchmark/core/ops/cpu/max_cpu.json";
    } else if (0 == strcmp(interface, "sum_cpu")){
        return "./lumos_t/benchmark/core/ops/cpu/sum_cpu.json";
    } else if (0 == strcmp(interface, "mean_cpu")){
        return "./lumos_t/benchmark/core/ops/cpu/mean_cpu.json";
    } else if (0 == strcmp(interface, "matrix_add_cpu")){
        return "./lumos_t/benchmark/core/ops/cpu/matrix_add_cpu.json";
    } else if (0 == strcmp(interface, "matrix_subtract_cpu")){
        return "./lumos_t/benchmark/core/ops/cpu/matrix_subtract_cpu.json";
    } else if (0 == strcmp(interface, "matrix_multiply_cpu")){
        return "./lumos_t/benchmark/core/ops/cpu/matrix_multiply_cpu.json";
    } else if (0 == strcmp(interface, "matrix_divide_cpu")){
        return "./lumos_t/benchmark/core/ops/cpu/matrix_divide_cpu.json";
    } else if (0 == strcmp(interface, "saxpy_cpu")){
        return "./lumos_t/benchmark/core/ops/cpu/fill_saxpy.json";
    } else if (0 == strcmp(interface, "sum_channel_cpu")){
        return "./lumos_t/benchmark/core/ops/cpu/sum_channel_cpu.json";
    } else if (0 == strcmp(interface, "gemm")){
        return "./lumos_t/benchmark/core/ops/gemm/gemm.json";
    } else if (0 == strcmp(interface, "gemm_nn")){
        return "./lumos_t/benchmark/core/ops/gemm/gemm_nn.json";
    } else if (0 == strcmp(interface, "gemm_nt")){
        return "./lumos_t/benchmark/core/ops/gemm/gemm_nt.json";
    } else if (0 == strcmp(interface, "gemm_tn")){
        return "./lumos_t/benchmark/core/ops/gemm/gemm_tn.json";
    } else if (0 == strcmp(interface, "gemm_tt")){
        return "./lumos_t/benchmark/core/ops/gemm/gemm_tt.json";
    } else if (0 == strcmp(interface, "im2col")){
        return "./lumos_t/benchmark/core/ops/im2col/im2col.json";
    } else if (0 == strcmp(interface, "col2im")){
        return "./lumos_t/benchmark/core/ops/im2col/col2im.json";
    } else if (0 == strcmp(interface, "avgpool")){
        return "./lumos_t/benchmark/core/ops/pooling/avgpool.json";
    } else if (0 == strcmp(interface, "maxpool")){
        return "./lumos_t/benchmark/core/ops/pooling/maxpool.json";
    } else if (0 == strcmp(interface, "avgpool_gradient")){
        return "./lumos_t/benchmark/core/ops/pooling/avgpool_gradient.json";
    } else if (0 == strcmp(interface, "maxpool_gradient")){
        return "./lumos_t/benchmark/core/ops/pooling/maxpool_gradient.json";
    } else if (0 == strcmp(interface, "forward_avgpool_layer")){
        return "./lumos_t/benchmark/core/graph/layer/avgpool_layer/forward_avgpool_layer.json";
    } else if (0 == strcmp(interface, "backward_avgpool_layer")){
        return "./lumos_t/benchmark/core/graph/layer/avgpool_layer/backward_avgpool_layer.json";
    } else if (0 == strcmp(interface, "forward_connect_layer")){
        return "./lumos_t/benchmark/core/graph/layer/connect_layer/forward_connect_layer.json";
    } else if (0 == strcmp(interface, "backward_connect_layer")){
        return "./lumos_t/benchmark/core/graph/layer/connect_layer/backward_connect_layer.json";
    } else if (0 == strcmp(interface, "forward_convolutional_layer")){
        return "./lumos_t/benchmark/core/graph/layer/convolutional_layer/forward_convolutional_layer.json";
    } else if (0 == strcmp(interface, "backward_convolutional_layer")){
        return "./lumos_t/benchmark/core/graph/layer/convolutional_layer/backward_convolutional_layer.json";
    } else if (0 == strcmp(interface, "forward_im2col_layer")){
        return "./lumos_t/benchmark/core/graph/layer/im2col_layer/forward_im2col_layer.json";
    } else if (0 == strcmp(interface, "backward_im2col_layer")){
        return "./lumos_t/benchmark/core/graph/layer/im2col_layer/backward_im2col_layer.json";
    } else if (0 == strcmp(interface, "forward_maxpool_layer")){
        return "./lumos_t/benchmark/core/graph/layer/maxpool_layer/forward_maxpool_layer.json";
    } else if (0 == strcmp(interface, "backward_maxpool_layer")){
        return "./lumos_t/benchmark/core/graph/layer/maxpool_layer/backward_maxpool_layer.json";
    } else if (0 == strcmp(interface, "forward_mse_layer")){
        return "./lumos_t/benchmark/core/graph/layer/mse_layer/forward_mse_layer.json";
    } else if (0 == strcmp(interface, "dropout_rand")){
        return "./lumos_t/benchmark/memory/dropout_rand.json";
    } else if (0 == strcmp(interface, "layer_delta")){
        return "./lumos_t/benchmark/memory/layer_delta.json";
    } else if (0 == strcmp(interface, "loss")){ //
        return "./lumos_t/benchmark/memory/loss.json";
    } else if (0 == strcmp(interface, "maxpool_index")){
        return "./lumos_t/benchmark/memory/maxpool_index.json";
    } else if (0 == strcmp(interface, "mean")){
        return "./lumos_t/benchmark/memory/mean.json";
    } else if (0 == strcmp(interface, "normalize_x_size")){
        return "./lumos_t/benchmark/memory/normalize_x_size.json";
    } else if (0 == strcmp(interface, "output")){
        return "./lumos_t/benchmark/memory/output.json";
    } else if (0 == strcmp(interface, "roll_mean")){
        return "./lumos_t/benchmark/memory/roll_mean.json";
    } else if (0 == strcmp(interface, "roll_variance")){
        return "./lumos_t/benchmark/memory/roll_variance.json";
    } else if (0 == strcmp(interface, "truth")){
        return "./lumos_t/benchmark/memory/truth.json";
    } else if (0 == strcmp(interface, "update_weights")){
        return "./lumos_t/benchmark/memory/update_weights.json";
    } else if (0 == strcmp(interface, "variance")){
        return "./lumos_t/benchmark/memory/variance.json";
    } else if (0 == strcmp(interface, "weights")){
        return "./lumos_t/benchmark/memory/weights.json";
    } else if (0 == strcmp(interface, "x_norm")){
        return "./lumos_t/benchmark/memory/x_norm.json";
    } else {
        fprintf(stderr, "  interface: %s is not in the testlist\n", interface);
    }
    return NULL;
}
