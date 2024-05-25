#include <iostream>
#include <fstream>
#include <cmath>
#include <sys/time.h>

using namespace std;

// function to add the elements of two arrays
void add(int n, float *x, float *y, float *z) {
    for (int i = 0; i < n; i++)
        z[i] = x[i] + y[i];
}

int main(int argc, char **argv) {

    int K = 1 << 20; // 1M elements
    int N;
    // Read command line argument
    if (argc == 2) {
        sscanf(argv[1], "%d", &K);
        N = K * 1000000;
    }
    else {
        printf("Usage: %s NumElements\n", argv[0]);
        exit(0);
    }

    float *x = new float[N];
    float *y = new float[N];
    float *z = new float[N];

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    double start, stop;
    struct timeval time;

    if (gettimeofday( &time, NULL ) < 0)
	    perror("start_timer,gettimeofday");

    start = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);

    // Run kernel on N elements on the CPU
    add(N, x, y, z);

    if (gettimeofday( &time, NULL ) < 0)
	    perror("stop_timer,gettimeofday");

    stop = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);

    cout << "Number of Elements: " << N << endl;
    cout << "Time Elapsed (CPU): " << stop - start << " secs" << endl;

    ofstream outfile("CPU_K_add.txt", ios::out | ios::app);
    if (outfile.is_open()) {
        outfile << K << " " << stop - start << endl; // Write to file
        outfile.close(); // Close file
    } else {
        cout << "Unable to open file for writing." << endl;
        return 1;
    }

    // Free memory
    delete[] x;
    delete[] y;
    delete[] z;

    return 0;
}
