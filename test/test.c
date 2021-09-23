#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    int x[] = {1, 0, -1, 1, 0, -1, 1, 0, -1};
    FILE *fp = fopen("./test.weights", "wb");
    fwrite(&x, 4, 9, fp);
    fclose(fp);

    fp = fopen("./test.weights", "rb");
    int *buffer = malloc(9*sizeof(int));
    fread(buffer, 4, 9, fp);
    for (int i = 0; i < 9; ++i){
        printf("%d\n", buffer[i]);
    }
    return 0;
}