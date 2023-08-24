#include "dataset.h"

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
    sess->dataset_num = format_str_line(tmp);
    sess->dataset_pathes = calloc(sess->dataset_num, sizeof(char*));
    for (int i = 0; i < sess->dataset_num; ++i){
        sess->dataset_pathes[i] = tmp+offset;
        offset += strlen(tmp+offset)+1;
    }
}

void bind_labelset(Session *sess, char *labelset_path_file)
{
    int offset = 0;
    FILE *fp = fopen(labelset_path_file, "r");
    fseek(fp, 0, SEEK_END);
    int file_size = ftell(fp);
    char *tmp = (char*)malloc(file_size * sizeof(char));
    memset(tmp, '\0', file_size * sizeof(char));
    fseek(fp, 0, SEEK_SET);
    fread(tmp, sizeof(char), file_size, fp);
    fclose(fp);
    sess->dataset_num = format_str_line(tmp);
    sess->labelset_pathes = calloc(sess->dataset_num, sizeof(char*));
    for (int i = 0; i < sess->dataset_num; ++i){
        sess->labelset_pathes[i] = tmp+offset;
        offset += strlen(tmp+offset)+1;
    }
}

int load_dataandlabel(Session *sess)
{
    int h, w, c;
    for (int i = 0; i < sess->subdivison; ++i){
        int index = (sess->index+i) % sess->dataset_num;
        char *path = sess->dataset_pathes[index];
        float *img = load_image_data(path, &h, &w, &c);
        resize_im(img, h, w, c, sess->height, sess->width, sess->input[i]);
        free(img);
        path = sess->labelset_pathes[index];
        sess->truth[i] = load_labels(path);
    }
    sess->index += sess->subdivison;
    if (sess->index >= sess->dataset_num){
        sess->index = 0;
        return 0;
    }
    return 1;
}
