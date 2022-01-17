#include "debug.h"

void save_data(float *data, int c, int h, int w, int batch, char *file)
{
    FILE *fp = fopen(file, "w");
    char buf[32];
    char *a = "  ";
    char b = '\n';
    char *d = "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    for (int bt = 0; bt < batch; ++bt){
        for (int k = 0; k < c; ++k){
            for (int i = 0; i < h; ++i){
                for (int j = 0; j < w; ++j){
                    sprintf(buf, "%f", data[bt*c*h*w+k*h*w+i*w+j]);
                    fputs(buf, fp);
                    fputs(a, fp);
                }
                fputc(b, fp);
            }
            fputc(b, fp);
            fputc(b, fp);
        }
        fputs(d, fp);
    }
    fclose(fp);
}