#include "utils.h"

#include "utils.h"

void fill_cpu_float(float *data, float x, int num)
{
    int i;
    for (i = 0; i < num; ++i){
        data[i] = x;
    }
}

float *flid_float_cpu(float *data, int h, int w, int c)
{
    int i, j, k;
    int rel_h, rel_l;
    float *output = malloc(h*w*c*sizeof(float));
    for (i = 0; i < c; ++i){
        for (j = 0; j < h; ++j){
            if (h%2==0){
                int middle_r_h = h/2;
                int middle_l_h = h/2 - 1;
                if (j < middle_l_h){
                    int dis = middle_l_h - j;
                    rel_h = middle_r_h + dis;
                }
                if (j > middle_r_h){
                    int dis = j - middle_r_h;
                    rel_h = middle_l_h - dis;
                }
                if (j == middle_l_h || j == middle_r_h) rel_h = j;
            }
            else{
                int middle_h = h/2;
                if (j < middle_h){
                    int dis = middle_h - j;
                    rel_h = middle_h + dis;
                }
                if (j > middle_h){
                    int dis = j - middle_h;
                    rel_h = middle_h - dis;
                }
                if (j == middle_h) rel_h = j;
            }
            for (k = 0; k < w; ++k){
                if (w%2 == 0){
                    int middle_d_w = w/2;
                    int middle_u_w = w/2 - 1;
                    if (k < middle_u_w){
                        int dis = middle_u_w - k;
                        rel_l = middle_d_w + dis;
                    }
                    if (k > middle_d_w){
                        int dis = k - middle_d_w;
                        rel_l = middle_u_w - dis;
                    }
                    if (k == middle_u_w || k == middle_d_w) rel_l = k;
                }
                else{
                    int middle_w = w/2;
                    if (k < middle_w){
                        int dis = middle_w - k;
                        rel_l = middle_w + dis;
                    }
                    if (k > middle_w){
                        int dis = k - middle_w;
                        rel_l = middle_w - dis;
                    }
                    if (k == middle_w) rel_l = k;
                }
                int index = (i*h + j)*w + k;
                int new_index = (i*h + rel_h)*w + rel_l;
                output[new_index] = data[index];
            }
        }
    }
    return output;
}

char *fgetl(FILE *fp)
{
    if(feof(fp)) return 0;
    char *line = malloc(512*sizeof(char));
    fgets(line, 512, fp);
    int len = strlen(line);
    line[len-1] = '\0';
    return line;
}

void strip(char *line)
{
    int len = strlen(line);
    int i;
    int offset = 0;
    for (i = 0; i < len; ++i){
        char c = line[i];
        if (c == ' ' || c == '\t' || c == '\n') ++offset;
        else line[i-offset] = c;
    }
    line[len-offset] = '\0';
}

char **split(char *line, char c, int *num)
{
    int len = strlen(line);
    int n = 0;
    int head = 0;
    int i = 0;
    for (i = 0; i < len; ++i){
        if (line[i] == c){
            if (i != head) n += 1;
            head = i+1;
        }
    }
    if (i != head) n += 1;
    char **res = malloc(n*sizeof(char *));
    num[0] = n;
    head = 0;
    n = 0;
    for (i = 0; i < len; ++i){
        if (line[i] == c){
            if (i != head) {
                char *A = malloc((i-head+1)*sizeof(char));
                memcpy(A, line+head, (i-head)*sizeof(char));
                A[i-head] = '\0';
                res[n] = A;
                n += 1;
            }
            head = i+1;
        }
    }
    if (i != head) {
        char *A = malloc((i-head+1)*sizeof(char));
        memcpy(A, line+head, (i-head)*sizeof(char));
        A[i-head] = '\0';
        res[n] = A;
    }
    return res;
}

char *int2str(int x)
{
    int a = x;
    int b;
    int n = 0;
    while (1)
    {
        if (a != 0) n += 1;
        else break;
        a = a / 10;
    }
    char *res = malloc(n*sizeof(char));
    a = x;
    for (int i = 0; i < n; ++i){
        b = a % 10;
        a = a / 10;
        res[i] = inten2str(b);
    }
    return res;
}

char *link_str(char *a, char *b)
{
    int len_a = strlen(a);
    int len_b = strlen(b);
    char *res = malloc((len_a+len_b)*sizeof(char));
    memcpy(res, a, len_a*sizeof(char));
    memcpy(res+len_a, b, len_b*sizeof(char));
    return res;
}

char *inten2str(int x)
{
    char *res = malloc(sizeof(char));
    if (x == 0) res[0] = '0';
    else if (x == 1) res[0] = '1';
    else if (x == 1) res[0] = '2';
    else if (x == 1) res[0] = '3';
    else if (x == 1) res[0] = '4';
    else if (x == 1) res[0] = '5';
    else if (x == 1) res[0] = '6';
    else if (x == 1) res[0] = '7';
    else if (x == 1) res[0] = '8';
    else if (x == 1) res[0] = '9';
    return res;
}