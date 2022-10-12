#include <stdio.h>
#include <stdlib.h>

#include "layer.h"
#include "avgpool_layer.h"
#include "utest.h"

void test_forward_avgpool_layer()
{
    test_run("test_forward_avgpool_layer");
    Layer *l;
    l = make_avgpool_layer(2);
    init_avgpool_layer(l, 4, 4, 2);
    float *input = calloc(l->inputs, sizeof(float));
    float *output = calloc(l->outputs, sizeof(float));
    float *workspace = calloc(l->workspace_size, sizeof(float));
    input[0] = 1;
    input[1] = 2;
    input[2] = 3;
    input[3] = 4;
    input[4] = 5;
    input[5] = 6;
    input[6] = 7;
    input[7] = 8;
    input[8] = 9;
    input[9] = 10;
    input[10] = 11;
    input[11] = 12;
    input[12] = 13;
    input[13] = 14;
    input[14] = 15;
    input[15] = 16;
    input[16] = 17;
    input[17] = 18;
    input[18] = 19;
    input[19] = 20;
    input[20] = 21;
    input[21] = 22;
    input[22] = 23;
    input[23] = 24;
    input[24] = 25;
    input[25] = 26;
    input[26] = 27;
    input[27] = 28;
    input[28] = 29;
    input[29] = 30;
    input[30] = 31;
    input[31] = 32;

    // input
    // 1  2  3  4      17 18 19 20
    // 5  6  7  8      21 22 23 24
    // 9  10 11 12     25 26 27 28
    // 13 14 15 16     29 30 31 32

    // output
    // 3.5   5.5       19.5  21.5
    // 11.5  13.5      27.5  29.5

    float truth[] = {3.5, 5.5, 11.5, 13.5, 19.5, 21.5, 27.5, 29.5};
    l->input = input;
    l->workspace = workspace;
    l->output = output;
    forward_avgpool_layer(*l, 1);
    for (int i = 0; i < l->outputs; ++i){
        if (fabs(output[i]-truth[i]) > 1e-6){
            printf("%d %f %f\n", i, output[i], truth[i]);
            test_res(1, "");
            return;
        }
    }
    test_res(0, "");
}

int main()
{
    test_forward_avgpool_layer();
}
