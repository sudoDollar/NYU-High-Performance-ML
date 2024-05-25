#include<stdio.h>
#include<math.h>
#include <stdlib.h>
#include <time.h>

float dpunroll(long N, float *pA, float *pB) {
    float R = 0.0;
    int j;
    for (j=0;j<N;j+=4)
        R += pA[j]*pB[j] + pA[j+1]*pB[j+1] + pA[j+2]*pB[j+2] + pA[j+3]*pB[j+3];
    return R;
}

int main(int argc, char *args[]) {
    long size = 1000, data = 0, flop = 0;
    int repetitions = 10;
    float *vectorA, *vectorB;
    struct timespec start, end;
    double total_time_usec = 0, mean_time, bandwidth, throughput, total_time_sec;
    float ans = 0.0;

    if(argc == 3) {
        char *endptr;
        size = strtol(args[1], &endptr, 10);
        repetitions = atoi(args[2]);
    }

    vectorA = (float*) calloc(size, sizeof(float));
    vectorB = (float*) calloc(size, sizeof(float));

    for(long i=0;i<size;i++) {
        vectorA[i] = 1.0;
        vectorB[i] = 1.0;
    }

    
    for(int j=0;j<repetitions;j++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        ans = dpunroll(size, vectorA, vectorB);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_usec = (((double)end.tv_sec *1000000 + (double)end.tv_nsec/1000) - ((double)start.tv_sec *1000000 + (double)start.tv_nsec/1000));
        if(j >= repetitions/2) {
            total_time_usec += time_usec;  //sec
            data += (size/4)*8*4; //bytes
            flop += (size/4)*8;
        }
    }

    printf("%f\n", ans);

    total_time_sec = total_time_usec / 1000000;
    
    mean_time = total_time_sec / (repetitions/2);  // sec
    bandwidth = data / (total_time_sec * 10e+9); // GB/sec
    throughput = flop / total_time_sec;  // flop/sec
    // printf("Total Time: %lf secs\n", total_time_sec);
    // printf("Arithemetic Intensity: %lf FLOP/byte\n\n", throughput/(bandwidth * 10e+9));    //To Do

    printf("N: %ld \t<T>: %.6lf sec \tB: %.3lf GB/sec \tF: %.3lf FLOP/sec \n", size, mean_time, bandwidth, throughput);

    return 0;
}