import numpy as np
import time
import sys

# for a simple loop
def dp(N,A,B):
    R = 0.0
    for j in range(0,N):
        R += A[j]*B[j]
    return R

size = 1000
repetitions = 10

n = len(sys.argv)
if n == 3:
    size = int(sys.argv[1])
    repetitions = int(sys.argv[2])

vector_A = np.ones(size, dtype=np.float32)
vector_B = np.ones(size, dtype=np.float32)
total_time_sec = 0
data = 0
flop = 0

for i in range(repetitions):
    start=time.monotonic()
    ans = dp(size, vector_A, vector_B)
    end = time.monotonic()
    if i >= repetitions/2:
        total_time_sec += (end - start)
        data += size * 2 * 4
        flop += size * 2

mean_time = total_time_sec / (repetitions/2)
bandwidth = data / (total_time_sec * 10e+9)
throughput = flop / total_time_sec  # flop/sec

print(ans)
print("N: {} \t<T>: {:.6f} sec \tB: {:.3f} GB/sec \tF: {:.3f} FLOP/sec".format(size, mean_time, bandwidth, throughput))

# print(throughput/(bandwidth * 10e+9))
