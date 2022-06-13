#include "layer.h"
#include "avgpool_layer.h"
#include "cfg_f.h"

#include <stdio.h>
#include <stdlib.h>

void test_make_avgpool_layer_by_arg()
{
    int ksize = 2;
    Layer l = make_avgpool_layer_by_arg(ksize);
    if (l.ksize == ksize){
        printf("[pass] ksize set success\n");
    }
}

void test_init_avgpool_layer()
{
    int ksize = 2;
    Layer l = make_avgpool_layer_by_arg(ksize);
    init_avgpool_layer(l, 2, 2, 1);
    if (l.input_h == 2 && l.input_w == 2 && l.input_c == 2){
        printf("[pass] input size correct\n");
    } else{
        printf([])
    }
    init_avgpool_layer(l, 2, 2, 2);
    init_avgpool_layer(l, 2, 2, 3);
    init_avgpool_layer(l, 2, 2, 4);

    init_avgpool_layer(l, 3, 3, 1);
    init_avgpool_layer(l, 3, 3, 3);
    init_avgpool_layer(l, 5, 5, 3);
    init_avgpool_layer(l, 5, 5, 3);
}