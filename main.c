#include <stdio.h>
#include <stdlib.h>

#include "session.h"
#include "manager.h"
#include "dispatch.h"
#include "graph.h"
#include "layer.h"
#include "convolutional_layer.h"
#include "connect_layer.h"
#include "im2col_layer.h"
#include "maxpool_layer.h"
#include "avgpool_layer.h"
#include "mse_layer.h"
#include "test.h"
#include "xor.h"

// void xor1_label2truth(char **label, float *truth)
// {
//     int x = atoi(label[0]);
//     one_hot_encoding(1, x, truth);
// }

// void xor1_process_test_information(char **label, float *truth, float *predict, float loss, char *data_path)
// {
//     fprintf(stderr, "Test Data Path: %s\n", data_path);
//     fprintf(stderr, "Label:   %s\n", label[0]);
//     fprintf(stderr, "Truth:   %f\n", truth[0]);
//     fprintf(stderr, "Predict: %f\n", predict[0]);
//     fprintf(stderr, "Loss:    %f\n\n", loss);
// }

int main(int argc, char **argv)
{
    xor();

    return 0;
}
