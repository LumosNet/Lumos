#ifndef CLUSTER_H
#define CLUSTER_H

#include "tensor.h"
#include "array.h"
#include "loss.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct middle_vt{
    char *label;
    Tensor *vt;
} middle_vt, MiddelVt;

typedef struct cluster_pile
{
    int k;
    MiddelVt **pile;
} cluster_pile, ClusterPile;

typedef struct cluster{
    Tensor *data;
    MiddelVt *mvt;
} cluster, Cluster;

typedef struct cluster_k{
    int k;
    Cluster **clusters;
} cluster_k, ClusterK;

#ifdef NUMPY
/* 无监督 每一行是一组数据*/
ClusterPile *k_means(Tensor *data, ClusterPile *initial, LossFunc lossfunc);


ClusterK *__classify(Tensor *data, ClusterPile *pile, LossFunc lossfunc);
ClusterPile *__middlevt(ClusterK *pile);
#endif

#ifdef  __cplusplus
}
#endif

#endif