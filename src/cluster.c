#include "cluster.h"

#ifdef NUMPY
// K_Means聚类算法
ClusterPile *k_means(Tensor *data, ClusterPile *initial, LossFunc lossfunc)
{
    ClusterK *new_classes;
    ClusterPile *new_pile = initial;
    while (1){
        ClusterK *classes = __classify(data, new_pile, lossfunc);
        new_classes = classes;
        ClusterPile *pile = __middlevt(new_classes);
        float loss = 0;
        for (int i = 0; i < initial->k; ++i){
            MiddelVt *vti = pile->pile[i];
            MiddelVt *vth = initial->pile[i];
            loss += lossfunc(vti->vt, vth->vt);
        }
        if ((loss / initial->k) <= 0.1) break;
        else new_pile = pile;
    }
    return new_pile;
}

// 根据聚类中心对数据重分类
ClusterK *__classify(Tensor *data, ClusterPile *pile, LossFunc lossfunc)
{
    ClusterK *clusterk = malloc(sizeof(ClusterK));
    clusterk->clusters = malloc(pile->k*sizeof(Cluster*));
    for (int i = 0; i < pile->k; ++i){
        Cluster *cluster = malloc(sizeof(Cluster));
        Tensor *x = array_x(0, data->size[0], 0);
        cluster->data = x;
        cluster->mvt = pile->pile[i];
        clusterk->clusters[i] = cluster;
    }
    for (int i = 0; i < data->size[1]; ++i){
        Tensor *x = row2Tensor(data, i+1);
        float loss = -1;
        int index = -1;
        for (int j = 0; j < pile->k; ++j){
            MiddelVt *vt = pile->pile[j];
            float lossk = lossfunc(x, vt->vt);
            if (lossk <= loss){
                loss = lossk;
                index = j;
            }
        }
        Cluster *belong = clusterk->clusters[index];
        Tensor *class = belong->data;
        Tensor *one = row2Tensor(data, i+1);
        insert_row(class, class->size[1]+1, one->data);
    }
    return clusterk;
}

// 计算聚类中心（中心向量）
ClusterPile *__middlevt(ClusterK *pile)
{
    ClusterPile *clusterpile = malloc(sizeof(ClusterPile));
    MiddelVt **mvt = malloc(pile->k*sizeof(MiddelVt*));
    clusterpile->k = pile->k;
    clusterpile->pile = mvt;
    for (int i = 0; i < pile->k; ++i){
        MiddelVt *vt = malloc(sizeof(MiddelVt));
        Cluster *cluster = pile->clusters[i];
        MiddelVt *ovt = cluster->mvt;
        vt->label = ovt->label;

        Tensor *data = cluster->data;
        Tensor *x = Tensor_x(data->size[0], 0, 1);
        Tensor *nvt = gemm(x, data);
        vt->vt = nvt;
        free_tensor(x);
    }
    return clusterpile;
}
#endif