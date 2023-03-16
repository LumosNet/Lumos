#include "call.h"

void call_ops(char *interface, void **params, void **ret)
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
    }
}

void call_graph(char *interface, void **params, void **ret)
{
    if (0 == strcmp(interface, "forward_avgpool_layer")){
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
    }
}
