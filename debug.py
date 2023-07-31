import time
first = 0

def lag_start():
    global first
    first = time.perf_counter()

def lag_end(string=''):
    global first
    print(string, time.perf_counter() - first)
    first = 0