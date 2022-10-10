#include "random.h"

float uniform_data(float a, float b, int *seed)
{
	float t;
	*seed = 2045.0 * (*seed) + 1;
	*seed = *seed - (*seed / 1048576) * 1048576;
	t = (*seed) / 1048576.0;
	t = a + (b - a) * t;
	return t;
}

float guass_data(float mean, float sigma, int *seed)
{
	int i;
	float x, y;
	for (x = 0, i = 0; i < 12; i++){
		x += uniform_data(0.0, 1.0, seed);
	}
	x = x - 6;
	y = mean + x * sigma;
	return y;
}

void guass_list(float mean, float sigma, int seed, int num, float *space)
{
    int sed[1] = {seed};
    for (int i = 0; i < num; ++i){
        space[i] = guass_data(mean, sigma, sed);
    }
}
