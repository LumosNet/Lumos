#include "cluster.h"

ClusterPile *k_means(Array *data, ClusterPile *initial, LossFunc lossfunc)
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

ClusterK *__classify(Array *data, ClusterPile *pile, LossFunc lossfunc)
{
    ClusterK *clusterk = malloc(sizeof(ClusterK));
    clusterk->clusters = malloc(pile->k*sizeof(Cluster*));
    for (int i = 0; i < pile->k; ++i){
        Cluster *cluster = malloc(sizeof(Cluster));
        Array *x = array_x(0, data->size[0], 0);
        cluster->data = x;
        cluster->mvt = pile->pile[i];
        clusterk->clusters[i] = cluster;
    }
    for (int i = 0; i < data->size[1]; ++i){
        Victor *x = row2Victor(data, i+1);
        float loss = -1;
        MiddelVt *best_mvt;
        int index = -1;
        for (int j = 0; j < pile->k; ++j){
            MiddelVt *vt = pile->pile[j];
            float lossk = lossfunc(x, vt->vt);
            if (lossk <= loss){
                loss = lossk;
                best_mvt = vt;
                index = j;
            }
        }
        Cluster *belong = clusterk->clusters[index];
        Array *class = belong->data;
        Victor *one = row2Victor(data, i+1);
        insert_row(class, class->size[1]+1, one->data);
    }
    return clusterk;
}

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

        Array *data = cluster->data;
        Victor *x = victor_x(data->size[0], 0, 1);
        Victor *nvt = gemm(x, data);
        vt->vt = nvt;
        del(x);
    }
    return clusterpile;
}