#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    int x[] = {1, 0, -1};
    int y[] = {2, 0, -2};
    FILE *fp = fopen("./test.weights", "wb");
    fwrite(&x, 4, 3, fp);
    fwrite(&y, 4, 3, fp);
    fclose(fp);

    fp = fopen("./test.weights", "rb");
    int *buffer1 = malloc(12*sizeof(int));
    // int *buffer2 = malloc(3*sizeof(int));
    fread(buffer1, 4, 12, fp);
    // fread(buffer2, 4, 3, fp);
    for (int i = 0; i < 12; ++i){
        printf("%d ", buffer1[i]);
    }
    printf("\n");
    return 0;
}

