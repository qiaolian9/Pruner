from tvm import te, auto_scheduler

@auto_scheduler.register_workload
def reduce_expr1(shape, dataType='float32'):
    r_idx = 1
    D1, D2 = shape
    
    K = shape[r_idx]
    A = te.placeholder((D1, D2), dtype=dataType, name="A")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((D1,), lambda d1: te.sum(A[d1, k], axis=k), name="C")
    return [A, C]

@auto_scheduler.register_workload
def reduce_expr2(shape, dataType='float32'):
    r_idx = 2
    D1, D2, D3 = shape

    K = shape[r_idx]
    A = te.placeholder((D1, D2, D3), dtype=dataType, name="A")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((D1, D2), lambda d1, d2: te.sum(A[d1, d2, k], axis=k), name="C")
    return [A, C]


@auto_scheduler.register_workload
def reduce_expr3(shape, dataType='float32'):
    r_idx1 = 2
    r_idx2 = 3
    D1, D2, D3, D4 = shape
    
    K1 = shape[r_idx1]
    K2 = shape[r_idx2]
    A = te.placeholder((D1, D2, D3, D4), dtype=dataType, name="A")
    k1 = te.reduce_axis((0, K1), name="k1")
    k2 = te.reduce_axis((0, K2), name="k2")
    C = te.compute((D1, D2), lambda d1, d2: te.sum(A[d1, d2, k1, k2], axis=[k1, k2]), name="C")
    return [A, C]

