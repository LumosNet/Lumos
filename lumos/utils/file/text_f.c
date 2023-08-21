#include "text_f.h"

char *fgetl(FILE *fp)
{
    if (feof(fp))
        return 0;
    char *line = malloc(512 * sizeof(char));
    fgets(line, 512, fp);
    int len = strlen(line);
    line[len - 1] = '\0';
    return line;
}

char **fgetls(FILE *fp)
{
    char **lines;
    char *line;
    int ln = 0;
    Lines *f_lines = malloc(sizeof(struct Lines));
    f_lines->next = NULL;
    Lines *f_head = f_lines;
    while ((line = fgetl(fp)) != 0)
    {
        if (line[0] == '\0')
            continue;
        Lines *f_line = malloc(sizeof(struct Lines));
        f_line->line = line;
        f_line->next = NULL;
        f_lines->next = f_line;
        f_lines = f_line;
        ln += 1;
    }
    lines = malloc((ln + 1) * sizeof(char *));
    ln = 1;
    while (f_head->next)
    {
        Lines *node = f_head->next;
        lines[ln] = node->line;
        f_head = f_head->next;
        ln += 1;
    }
    lines[0] = int2str(ln - 1);
    return lines;
}

void fputl(FILE *fp, char *line)
{
    fputs(line, fp);
    fputc('\n', fp);
}

void fputls(FILE *fp, char **lines, int n)
{
    for (int i = 0; i < n; ++i)
    {
        fputl(fp, lines[i]);
    }
}

// char **load_label_txt(char *path, int num)
// {
//     char **label = calloc(num, sizeof(char *));
//     int label_offset = 0;
//     FILE *fp = fopen(path, "r");
//     if (fp == NULL){
//         fprintf(stderr, "\nerror file %s can not open\n", path);
//         abort();
//     }
//     char *line = fgetl(fp);
//     fclose(fp);
//     int n[1];
//     char **nodes = split(line, ' ', n);
//     for (int k = 0; k < n[0]; ++k)
//     {
//         strip(nodes[k], ' ');
//         if (label_offset >= num)
//             return label;
//         label[label_offset] = nodes[k];
//         label_offset += 1;
//     }

    
//     return label;
// }

void **load_label_txt(char *path)
{
    char *tmp = fget(path);
    int *index = split(tmp, ' ');
    void **res = malloc(2*sizeof(void*));
    res[0] = (void*)index;
    res[1] = (void*)tmp;
    return res;
}

char *fget(char *file)
{
    FILE *fp = fopen(file, "r");
    if (fp == NULL) {
        fprintf(stderr, "\nfopen error: %s is not exist\n", file);
        abort();
    }
    fseek(fp, 0, SEEK_END);
    int file_size = ftell(fp);
    char *tmp = (char*)malloc(file_size * sizeof(char));
    memset(tmp, '\0', file_size * sizeof(char));
    fseek(fp, 0, SEEK_SET);
    fread(tmp, sizeof(char), file_size, fp);
    fclose(fp);
    return tmp;
}

int lines_num(char *tmp)
{
    int res = 0;
    int len = strlen(tmp);
    int flag = 0;
    for (int i = 0; i < len; ++i){
        if (tmp[i] == '\r' || tmp[i] == '\n'){
            if (flag) res += 1;
            flag = 0;
        } else if (tmp[i] == ' ') {
            continue;
        } else {
            flag = 1;
        }
    }
    return res;
}

int format_str(char *tmp)
{
    int res = 0;
    int len = strlen(tmp);
    int flag = 0;
    int meet = 0;
    int n = 0;
    for (int i = 0; i < len; ++i){
        if (tmp[i] == '\r' || tmp[i] == '\n') flag = 1;
        if (tmp[i] == '\r' || tmp[i] == '\n' || tmp[i] == '\t' || tmp[i] == ' '){
            tmp[i] = '\0';
            n += 1;
        } else {
            if (n != 0){
                if (flag && meet){
                    char *head = tmp+i-n;
                    head[0] = '\0';
                    res += 1;
                    n -= 1;
                }
                memcpy(tmp+i-n, tmp+i, (len-i)*sizeof(char));
                i -= n;
            }
            n = 0;
            flag = 0;
            meet = 1;
        }
    }
    if (meet) res += 1;
    return res;
}
