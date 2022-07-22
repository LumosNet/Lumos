#include "cfg_f.h"

CFGParam *make_cfg_param(char *param_line)
{
    CFGParam *cfg_param = malloc(sizeof(struct CFGParam));
    int *num = malloc(sizeof(int));
    char **param = split(param_line, '=', num);
    strip(param[0], ' ');
    strip(param[1], ' ');
    cfg_param->key = param[0];
    cfg_param->val = param[1];
    cfg_param->next = NULL;
    return cfg_param;
}

CFGPiece *make_cfg_piece(char *name_line)
{
    CFGPiece *cfg_piece = malloc(sizeof(struct CFGPiece));
    cfg_piece->param_num = 0;
    cfg_piece->name = malloc((strlen(name_line)-2)*sizeof(char));
    memcpy(cfg_piece->name, name_line+1, (strlen(name_line)-2)*sizeof(char));
    cfg_piece->name[strlen(name_line)-2] = '\0';
    cfg_piece->next = NULL;
    return cfg_piece;
}

void insert_cfg_params(CFGParams *cfg_params, CFGParam *cfg_param)
{
    if (cfg_params->head){
        CFGParam *tail = cfg_params->tail;
        tail->next = cfg_param;
        cfg_params->tail = cfg_param;
    } else{
        cfg_params->head = cfg_param;
        cfg_params->tail = cfg_param;
    }
}

void insert_cfg_pieces(CFGPieces *cfg_pieces, CFGPiece *cfg_piece)
{
    if (cfg_pieces->head){
        CFGPiece *tail = cfg_pieces->tail;
        tail->next = cfg_piece;
        cfg_pieces->tail = cfg_piece;
    } else{
        cfg_pieces->head = cfg_piece;
        cfg_pieces->tail = cfg_piece;
    }
}

CFG *load_conf_cfg(char *file)
{
    FILE *fp = fopen(file, "r");
    char **lines = fgetls(fp);
    char *line;
    int n_line = atoi(lines[0]);
    CFG *cfg = malloc(sizeof(struct CFG));
    CFGPieces *cfg_pieces = malloc(sizeof(struct CFGPieces));
    CFGPiece *cfg_piece;
    CFGParams *cfg_params;
    CFGParam *cfg_param;
    cfg_pieces->head = NULL;
    cfg_pieces->tail = NULL;
    cfg->pieces = cfg_pieces;
    cfg->piece_num = 0;
    for (int i = 0; i < n_line; ++i){
        line = lines[i+1];
        switch (line[0]){
            case '[':
                cfg_piece = make_cfg_piece(line);
                cfg_params = malloc(sizeof(struct CFGParams));
                cfg_params->head = NULL;
                cfg_params->tail = NULL;
                cfg_piece->params = cfg_params;
                insert_cfg_pieces(cfg_pieces, cfg_piece);
                cfg->piece_num += 1;
                break;
            case '#':
                break;
            default:
                cfg_param = make_cfg_param(line);
                insert_cfg_params(cfg_params, cfg_param);
                cfg_piece->param_num += 1;
        }
    }
    return cfg;
}

int get_piece_param_n(CFGPiece *cfg_piece)
{
    return cfg_piece->param_num;
}

int get_cfg_piece_n(CFG *cfg)
{
    return cfg->piece_num;
}

char *get_piece_name(CFGPiece *cfg_piece)
{
    return cfg_piece->name;
}

char *get_param_by_key(CFGPiece *cfg_piece, char *key)
{
    CFGParams *params = cfg_piece->params;
    CFGParam *param = params->head;
    char *val = NULL;
    while (param){
        if (0 == strcmp(key, param->key)){
            val = param->val;
            break;
        }
    }
    return val;
}

CFGPiece *get_piece(CFG *cfg, int index)
{
    CFGPieces *pieces = cfg->pieces;
    CFGPiece *piece = pieces->head;
    CFGPiece *res = NULL;
    int n = 0;
    while (piece){
        if (n+1 == index){
            res = piece;
            break;
        }
        piece = piece->next;
        n++;
    }
    return res;
}

CFGParam *get_param(CFGPiece *cfg_piece, int index)
{
    CFGParams *params = cfg_piece->params;
    CFGParam *param = params->head;
    CFGParam *res;
    int n = 0;
    while (param){
        if (n+1 == index){
            res = param;
            break;
        }
        param = param->next;
        n++;
    }
    return res;
}
