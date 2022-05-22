#include <stdio.h>
#include <stdlib.h>

#include "cfg_f.h"

int main(int argc, char **argv)
{
    CFG *cfg = load_conf_cfg("./cfg/xor.cfg");
    // printf("piece_num: %d\n", cfg->piece_num);
    // CFGPieces *pieces = cfg->pieces;
    // CFGPiece *piece = pieces->head;
    // CFGParams *cfgparams;
    // while (piece){
    //     printf("param_num: %d\n", piece->param_num);
    //     printf("piece name: %s\n", piece->name);
    //     cfgparams = piece->params;
    //     CFGParam *cfgparam = cfgparams->head;
    //     while (cfgparam){
    //         printf("%s = %s\n", cfgparam->key, cfgparam->val);
    //         cfgparam = cfgparam->next;
    //     }
    //     piece = piece->next;
    // }
    int n = get_cfg_piece_n(cfg);
    printf("%d\n", n);
    CFGPiece *cfg_piece = get_piece(cfg, 2);
    int p_n = get_piece_param_n(cfg_piece);
    printf("%d\n", p_n);
    char *name = get_piece_name(cfg_piece);
    printf("%s\n", name);
    char *val = get_param_by_key(cfg_piece, "flag");
    printf("%s\n", val);
    CFGParam *cfg_param = get_param(cfg_piece, 1);
    printf("%s = %s\n", cfg_param->key, cfg_param->val);
}