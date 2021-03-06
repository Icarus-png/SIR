from numba import jit, cuda
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer


@cuda.jit  # (device=True)
def sir_kernel(s, i, r, b, g):
    """
    Model formula: https://www.maa.org/press/periodicals/loci/joma/the-sir-model-for-spread-of-disease-the-differential-equation-model
    :param g: gamma
    :param b: beta
    :param s: Initial Susceptible
    :param i: Initial Infected
    :param r: Initial Recovered
    :return: N/a
    """
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    block_size = cuda.blockDim.x
    grid_size = cuda.gridDim.x
    startt = tx + ty * block_size
    stride = block_size * grid_size

    for x in range(len(s) - 1):
        # dSdt = -beta * s(t) * i(t)
        s[x + 1] = s[x] - (b * s[x] * i[x])
        # didt = beta * s(t) * i(t) - gamma * i(t)
        i[x + 1] = i[x] + (b * s[x] * i[x] - g * i[x])
        # drdt = gamma * i(t)
        r[x + 1] = r[x] + (g * i[x])
        cuda.syncthreads()


N = 1000  # Total Population
I0 = 10  # Initial number of infected
R0 = 0  # Initial number of Recovered/immune
S0 = N - I0 - R0  # Initial number of susceptible people
beta = 0.4  # Contact rate
gamma = .12  # recovery rate
time = 100  # Total sim time in days
# Create float arrays to store populations over time
susceptible, infected, recovered = np.zeros([time]), np.zeros([time]), np.zeros([time])

# Calculate initial percentage (compatible with numpy) and set to first variable
# Reference: https://airodoctor.com/us/resources/calculation/infectious-disease-and-epidemic-calculator-sir-model/
susceptible[0] = S0 / N
infected[0] = I0 / N
recovered[0] = R0 / N

threads_per_block = 128
blocks_per_grid = 32

s_device = cuda.to_device(susceptible)
i_device = cuda.to_device(infected)
r_device = cuda.to_device(recovered)

start = timer()
cuda.synchronize()
sir_kernel[blocks_per_grid, threads_per_block](s_device, i_device, r_device, beta, gamma)
cuda.synchronize()
end = timer() - start
print(end)

susceptible = s_device.copy_to_host()
infected = i_device.copy_to_host()
recovered = r_device.copy_to_host()

fig, axes = plt.subplots(figsize=(10, 6))
axes.plot(susceptible, c='b', alpha=0.5, lw=3, label='Susceptible')
axes.plot(infected, c='r', alpha=0.5, lw=3, label='Infected')
axes.plot(recovered, c='g', alpha=0.5, lw=3, label='Recovered')
axes.set_xlabel('Time (Days)', fontsize=16)
axes.set_ylabel('Population Percentage', fontsize=16)
axes.grid(1)
plt.title('Parallel SIR Data')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='best')
plt.show()

