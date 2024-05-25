import matplotlib.pyplot as plt

k_cpu = []
t_cpu = []

with open('CPU_K_add.txt', 'r') as file:
    for line in file:
        val = line.split(" ")
        k_cpu.append(float(val[0]))
        t_cpu.append(float(val[1]))


#Non Unified Memory => 1 Block, 1 Thread
k = []
t = []

with open('GPU_K_add_1_1.txt', 'r') as file:
    for line in file:
        val = line.split(" ")
        k.append(float(val[0]))
        t.append(float(val[1]))

fig = plt.figure()
plt.plot(k, t, label = "GPU")
plt.scatter(k,t)
plt.plot(k_cpu, t_cpu, color='r', label = "CPU")
plt.scatter(k_cpu, t_cpu, color='r')
plt.xlabel("K (in millions)")
plt.ylabel("Time")
plt.yscale("log")
plt.xscale("log")
plt.legend()

plt.title("GPU (1 Block, 1 Thread) (Non Unified Memory)")
plt.savefig("gpu_k_1_1.png")

#Non Unified Memory => 1 Block, 256 Threads
k = []
t = []

with open('GPU_K_add_1_256.txt', 'r') as file:
    for line in file:
        val = line.split(" ")
        k.append(float(val[0]))
        t.append(float(val[1]))

fig = plt.figure()
plt.plot(k, t, label = "GPU")
plt.scatter(k,t)
plt.plot(k_cpu, t_cpu, color='r', label = "CPU")
plt.scatter(k_cpu, t_cpu, color='r')
plt.xlabel("K (in millions)")
plt.ylabel("Time")
plt.yscale("log")
plt.xscale("log")
plt.legend()

plt.title("GPU (1 Block, 256 Thread) (Non Unified Memory)")
plt.savefig("gpu_k_1_256.png")

#Non Unified Memory => Variable Blocks, 256 Threads
k = []
t = []

with open('GPU_K_add_-1_256.txt', 'r') as file:
    for line in file:
        val = line.split(" ")
        k.append(float(val[0]))
        t.append(float(val[1]))

fig = plt.figure()
plt.plot(k, t, label = "GPU")
plt.scatter(k,t)
plt.plot(k_cpu, t_cpu, color='r', label = "CPU")
plt.scatter(k_cpu, t_cpu, color='r')
plt.xlabel("K (in millions)")
plt.ylabel("Time")
plt.yscale("log")
plt.xscale("log")
plt.legend()

plt.title("GPU (var Block, 256 Thread) (Non Unified Memory)")
plt.savefig("gpu_k_-1_256.png")


#Unified Memory => 1 Block, 1 Thread
k = []
t = []

with open('GPU_unified_K_add_1_1.txt', 'r') as file:
    for line in file:
        val = line.split(" ")
        k.append(float(val[0]))
        t.append(float(val[1]))

fig = plt.figure()
plt.plot(k, t, label = "GPU")
plt.scatter(k,t)
plt.plot(k_cpu, t_cpu, color='r', label = "CPU")
plt.scatter(k_cpu, t_cpu, color='r')
plt.xlabel("K (in millions)")
plt.ylabel("Time")
plt.yscale("log")
plt.xscale("log")
plt.legend()

plt.title("GPU (1 Block, 1 Thread) (Unified Memory)")
plt.savefig("gpu_k_unified_1_1.png")


#Unified Memory => 1 Block, 256 Threads
k = []
t = []

with open('GPU_unified_K_add_1_256.txt', 'r') as file:
    for line in file:
        val = line.split(" ")
        k.append(float(val[0]))
        t.append(float(val[1]))

fig = plt.figure()
plt.plot(k, t, label = "GPU")
plt.scatter(k,t)
plt.plot(k_cpu, t_cpu, color='r', label = "CPU")
plt.scatter(k_cpu, t_cpu, color='r')
plt.xlabel("K (in millions)")
plt.ylabel("Time")
plt.yscale("log")
plt.xscale("log")
plt.legend()

plt.title("GPU (1 Block, 256 Threads) (Unified Memory)")
plt.savefig("gpu_k_unified_1_256.png")


#Unified Memory => Variable Blocks, 256 Threads
k = []
t = []

with open('GPU_unified_K_add_-1_256.txt', 'r') as file:
    for line in file:
        val = line.split(" ")
        k.append(float(val[0]))
        t.append(float(val[1]))

fig = plt.figure()
plt.plot(k, t, label = "GPU")
plt.scatter(k,t)
plt.plot(k_cpu, t_cpu, color='r', label = "CPU")
plt.scatter(k_cpu, t_cpu, color='r')
plt.xlabel("K (in millions)")
plt.ylabel("Time")
plt.yscale("log")
plt.xscale("log")
plt.legend()

plt.title("GPU (var Block, 256 Threads) (Unified Memory)")
plt.savefig("gpu_k_unified_-1_256.png")