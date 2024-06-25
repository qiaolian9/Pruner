from tvm import te, auto_scheduler
import os

CNHW = (os.getenv('CONV_LAYOUT')=="CNHW")

@auto_scheduler.register_workload
def matmul(shape, dataType="float32"):
    M, N, K = shape
    A = te.placeholder((M, K), dtype=dataType, name="A")
    B = te.placeholder((K, N), dtype=dataType, name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((M, N), lambda x, y: te.sum((A[x, k]) * B[k, y], axis=k), name='compute')
    return [A, B, C]

@auto_scheduler.register_workload
def dense(shape, dataType="float32"):
    M, N, K = shape
    A = te.placeholder((M, K), dtype=dataType, name="A")
    B = te.placeholder((K, N), dtype=dataType, name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((M, N), lambda y, x: te.sum((A[y, k]) * B[k, x], axis=k), name='compute')
    D = te.compute((M, N), lambda i, j: C[i, j] + 1, name='D')
    return [A, B, D]

@auto_scheduler.register_workload
def batch_matmul(shape, dataType="float32"):
    BC, M, N, K = shape
    A = te.placeholder((BC, M, K), dtype=dataType, name="A")
    B = te.placeholder((BC, K, N), dtype=dataType, name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((BC, M, N), lambda b, x, y: te.sum(A[b, x, k] * B[b, k, y], axis=k))
    return [A, B, C]
