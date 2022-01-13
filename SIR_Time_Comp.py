from numba import jit, cuda
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

times = [100, 1000, 10000, 100000, 1000000, 10000000, 100000000]


def sir_model(s, i, r, b, g):
    tme = len(s)
    for t in range(tme - 1):
        # dSdt = -beta * s(t) * i(t)
        s[t + 1] = s[t] - b * s[t] * i[t]
        # didt = beta * s(t) * i(t) - gamma * i(t)
        i[t + 1] = i[t] + b * s[t] * i[t] - g * i[t]
        # drdt = gamma * i(t)
        r[t + 1] = r[t] + g * i[t]


@cuda.jit  # (device=True)
def sir_kernel(s, i, r, b, g):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    block_size = cuda.blockDim.x
    grid_size = cuda.gridDim.x
    startt = tx + ty * block_size
    stride = block_size * grid_size

    tme = len(s)
    for t in range(tme - 1):
        s[t + 1] = s[t] - b * s[t] * i[t]
        i[t + 1] = i[t] + b * s[t] * i[t] - g * i[t]
        r[t + 1] = r[t] + g * i[t]
        cuda.syncthreads()


def run_base(tme):
    N = 1000  # Total Population
    I0 = 10  # Initial number of infected
    R0 = 0  # Initial number of Recovered/immune
    S0 = N - I0 - R0  # Initial number of susceptible people
    beta = 0.4  # Contact rate
    gamma = .12  # recovery rate
    time = tme  # Total sim time in days
    # Create float arrays to store populations over time
    susceptible, infected, recovered = np.zeros([time]), np.zeros([time]), np.zeros([time])

    susceptible[0] = S0 / N
    infected[0] = I0 / N
    recovered[0] = R0 / N

    start = timer()
    sir_model(susceptible, infected, recovered, beta, gamma)
    tstop = timer() - start
    print("serial time: ", tstop)
    return tstop


def run_parallel(tme):
    N = 1000  # Total Population
    I0 = 10  # Initial number of infected
    R0 = 0  # Initial number of Recovered/immune
    S0 = N - I0 - R0  # Initial number of susceptible people
    beta = 0.4  # Contact rate
    gamma = .12  # recovery rate
    time = tme  # Total sim time in days
    susceptible, infected, recovered = np.zeros([time]), np.zeros([time]), np.zeros([time])

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
    print("parallel time: ", end)
    return end

    susceptible = s_device.copy_to_host()
    infected = i_device.copy_to_host()
    recovered = r_device.copy_to_host()


if __name__ == '__main__':
    base_times = np.zeros(len(times))
    par_times = np.zeros(len(times))
    for t in range(len(times)):
        base_times[t] = run_base(times[t])
        par_times[t] = run_parallel(times[t])

    x = list(range(len(times)))
    fig, axes = plt.subplots(figsize=(10, 6))
    axes.plot(base_times, c='b', alpha=0.5, lw=3, label='serial')
    axes.plot(par_times, c='g', alpha=0.5, lw=3, label='parallel')
    axes.set_xlabel('SIR Time(Days)', fontsize=16)
    axes.set_ylabel('Execution Time (sec)', fontsize=16)
    axes.grid(1)
    plt.title('Serial v Parallel Execution Times (large)', fontsize=16)
    plt.xticks(x, times, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='best')
    plt.show()
