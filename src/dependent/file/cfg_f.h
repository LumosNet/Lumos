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

typedef struct CFGParam{
    char *key;
    char *val;
    struct CFGParam *next;
} CFGParam;

typedef struct CFGParams{
    struct CFGParam *head;
    struct CFGParam *tail;
} CFGParams;

typedef struct CFGPiece{
    int param_num;
    char *name;
    struct CFGParams *params;
    struct CFGPiece *next;
} CFGPiece;

typedef struct CFGPieces{
    struct CFGPiece *head;
    struct CFGPiece *tail;
} CFGPieces;

typedef struct CFG{
    int piece_num;
    struct CFGPieces *pieces;
} CFG;

CFGParam *make_cfg_param(char *param_line);
CFGPiece *make_cfg_piece(char *name_line);
void insert_cfg_params(CFGParams *cfg_params, CFGParam *cfg_param);
void insert_cfg_pieces(CFGPieces *cfg_pieces, CFGPiece *cfg_piece);
CFG *load_conf_cfg(char *file);

int get_piece_param_n(CFGPiece *cfg_piece);
int get_cfg_piece_n(CFG *cfg);
char *get_piece_name(CFGPiece *cfg_piece);
char *get_param_by_key(CFGPiece *cfg_piece, char *key);
CFGPiece *get_piece(CFG *cfg, int index);
CFGParam *get_param(CFGPiece *cfg_piece, int index);

#ifdef __cplusplus
}
#endif

#endif