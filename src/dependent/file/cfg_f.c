#include "cfg_f.h"

void _insert_cfg_piece(struct CFG *cfg, struct CfgPiece *cfg_piece)
{

}

CFG *load_file_cfg(char *file)
{
    FILE *fp = fopen(file, "r");
    char **lines = fgetls(fp);
    char *line;
    int n_line = atoi(lines[0]);
    CFG *cfg = malloc(sizeof(struct CFG));
    CFGPieces *cfg_pieces = malloc(sizeof(struct CFGPieces));
    cfg_pieces->head = NULL;
    cfg_pieces->tail = NULL;
    cfg->pieces = cfg_pieces;
    cfg->piece_num = 0;
    CFGParams *cfg_params;
    for (int i = 0; i < n_line; ++i){
        line = lines[i+1];
        switch (line[0]){
            case '[':
                CfgPiece *cfg_piece = malloc(sizeof(struct CFGPiece));
                cfg_piece->param_num = 0;
                cfg_piece->name = malloc((strlen(line)-2)*sizeof(char));
                memcpy(cfg_piece->name, line+1, (strlen(line)-2)*sizeof(char));
                cfg_piece->name[strlen(line)-2] = '\0';
                cfg_params = malloc(sizeof(struct CfgParams));
                cfg_params->head = NULL;
                cfg_params->tail = NULL;
                cfg_piece->params = cfg_params;
                cfg_piece->next = NULL;

                if (cfg_pieces->head){
                    CfgPiece *tail = cfg_pieces->tail;
                    tail->next = cfg_piece;
                    cfg_pieces->tail = cfg_piece;
                } else{
                    cfg_pieces->head = cfg_piece;
                    cfg_pieces->tail = cfg_piece;
                }
                cfg->piece_num += 1;
                break;
            case '#':
                break;
            default:
                CfgParam *cfg_param = malloc(sizeof(struct CfgParam));
                int *num = malloc(sizeof(int));
                char **param = split(line, '=', num);
                strip(param[0], ' ');
                strip(param[1], ' ');
                cfg_param->key = param[0];
                cfg_param->val = param[1];

                if (cfg_params->head){
                    CfgParam *tail = cfg_params->tail;
                    tail->next = cfg_param;
                    cfg_params->tail = cfg_param;
                } else{
                    cfg_params->head = cfg_param;
                    cfg_params->tail = cfg_param;
                }
                cfg_piece->params_num += 1;
        }
    }
}