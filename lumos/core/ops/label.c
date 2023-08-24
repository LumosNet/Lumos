#include "label.h"

float *load_labels(char *path)
{
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        fprintf(stderr, "\nfopen error: %s is not exist\n", path);
        abort();
    }
    fseek(fp, 0, SEEK_END);
    int file_size = ftell(fp);
    char *tmp = (char*)malloc(file_size * sizeof(char));
    memset(tmp, '\0', file_size * sizeof(char));
    fseek(fp, 0, SEEK_SET);
    fread(tmp, sizeof(char), file_size, fp);
    fclose(fp);
    int num = format_str_space(tmp);
    float *labels = calloc(num, sizeof(float));
    int offset = 0;
    for (int i = 0; i < num; ++i){
        char *label = tmp + offset;
        labels[i] = atof(label);
        offset += strlen(tmp+offset)+1;
    }
    free(tmp);
    return labels;
}
