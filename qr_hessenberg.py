import datetime as time
import matplotlib.pyplot as plt
import numpy as np


STEPS = 1
START = 1
END = 50

# plot

def export_plot():
    t = np.arange(START, END, STEPS)
    plt.plot(t, assemble_matrix.execution_times, label="Matrix Assembly")
    plt.plot(t, backwards_sub.execution_times, label="Backwards Substitution")
    plt.plot(t, qr.execution_times, label="QR-Decomposition")
    plt.plot(t, matrix_prod.execution_times, label="Matrix Product")
    plt.legend()
    plt.show()

# calculations

def timing_decorator(func):
    execution_times = []

    def wrapper(*args, **kwargs):
        start_time = time.datetime.now()
        result = func(*args, **kwargs)
        end_time = time.datetime.now()
        execution_time = (end_time - start_time).total_seconds() * 1000
        execution_times.append(execution_time)
        return result

    wrapper.execution_times = execution_times
    return wrapper

def flush_all():
    solve.execution_times = []
    qr.execution_times = []
    matrix_prod.execution_times = []
    assemble_matrix.execution_times = []

@timing_decorator
def calculate_times():

    # actual calc
    results = []
    percent = 0
    for i in range(START, END, STEPS):
        A, b = assemble_matrix(i)
        x = solve(A, b)
        results.append(x)
        new_percent = (((i - START)/STEPS)/((END-START)/STEPS))*100
        if new_percent >= (percent + 1):
            percent = new_percent
            print(f"{round(new_percent)}%")
    export_plot()

@timing_decorator
def solve(A, b):

    Q, R = qr(A)
    q_transposed: np.ndarray = np.transpose(Q)
    c = matrix_prod(q_transposed, b)
    x = backwards_sub(R, c)
    return x

@timing_decorator
def qr(A):
    return np.linalg.qr(A)

@timing_decorator
def backwards_sub(A, b):
    n = len(A)
    x = [0 for i in range(n)]
    for k in reversed(range(n)):
        sum = 0
        for i in range(k+1, n):
            sum = sum + (A[k, i] * x[i])
        x[k] = (b[k] - sum)/A[k, k]
    return np.array(x)


@timing_decorator
def matrix_prod(a, b):
    return a.dot(b)

@timing_decorator
def assemble_matrix(N: int):
    # A
    arr = []
    for m in range(1, N+1):
        row = []
        for n in range(1, N+1):
            value = 0
            if m <= n+1:
                value = 1/(m+n-1)
            row.append(value)
        arr.append(row)
    # b
    vector = []
    for m in range(0, N):
        sum = 0
        for n in range(0, N):
            sum = sum + arr[m][n]
        vector.append(sum)
    return np.array(arr), np.array(vector)
