#include <stdio.h>
#include "ex4.h"

#define SAMPLE_RATE 1.0

int main() {
    int num_samples = sizeof(DefaultProfile)/sizeof(double);
    double velocity = 0.0;
    double position = 0.0;

    for (int i = 0; i < num_samples - 1; i++) {
        double interval = (double)(i + 1) / SAMPLE_RATE - (double)i / SAMPLE_RATE;
        double area = (DefaultProfile[i] + DefaultProfile[i + 1]) / 2 * interval;
        velocity += area;
        position += velocity * interval + area / 2;
    }

    printf("Final velocity: %lf\n", velocity);
    printf("Final position: %lf\n", position);

    return 0;
}
