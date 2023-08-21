#include "input.h"

void bind_dataset(Session *sess, char *dataset_path_file)
{
    int offset = 0;
    FILE *fp = fopen(dataset_path_file, "r");
    fseek(fp, 0, SEEK_END);
    int file_size = ftell(fp);
    char *tmp = (char*)malloc(file_size * sizeof(char));
    memset(tmp, '\0', file_size * sizeof(char));
    fseek(fp, 0, SEEK_SET);
    fread(tmp, sizeof(char), file_size, fp);
    fclose(fp);
    sess->dataset_num = format_str(tmp);
    sess->dataset_pathes = calloc(sess->dataset_num, sizeof(char*));
    for (int i = 0; i < sess->dataset_num; ++i){
        sess->dataset_pathes[i] = tmp+offset;
        offset += strlen(tmp+offset)+1;
    }
}
