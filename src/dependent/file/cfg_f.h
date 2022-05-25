#ifndef CFG_F_H
#define CFG_F_H

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CfgParam{
    char *key;
    char *value;
} CfgParam;

typedef struct CfgParams{
    struct CfgParam *front;
    struct CfgParam *next;
} CfgParams;

typedef struct CfgPiece{
    int param_num;
    char *name;
    struct CfgParams *params;
} CfgPiece;

typedef struct CFG{
    int piece_num;
    struct CfgPiece *pieces;
} CFG;

CFG *load_file_cfg(char *file);


#ifdef __cplusplus
}
#endif

#endif