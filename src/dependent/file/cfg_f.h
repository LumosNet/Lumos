#ifndef CFG_F_H
#define CFG_F_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "text_f.h"
#include "str_ops.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CfgParam{
    char *key;
    char *val;
    struct CfgParam *next;
} CfgParam;

typedef struct CfgParams{
    struct CfgParam *head;
    struct CfgParam *tail;
} CfgParams;

typedef struct CfgPiece{
    int param_num;
    char *name;
    struct CfgParams *params;
    struct CfgPiece *next;
} CfgPiece;

typedef struct CFGPieces{
    struct CfgPiece *head;
    struct CfgPiece *tail;
} CFGPieces;

typedef struct CFG{
    int piece_num;
    struct CFGPieces *pieces;
} CFG;

CfgParam *make_cfg_param(char *param_line);
CfgPiece *make_cfg_piece(char *name_line);
void insert_cfg_params(CfgParams *cfg_params, CfgParam *cfg_param);
void insert_cfg_pieces(CFGPieces *cfg_pieces, CfgPiece *cfg_piece);
CFG *load_file_cfg(char *file);

#ifdef __cplusplus
}
#endif

#endif