
from time import perf_counter

def benchmark(fn,repeats=3):
    
    times=[]
    for _ in range(repeats):
        start=perf_counter();fn();times.append(perf_counter()-start)
    return min(times)
