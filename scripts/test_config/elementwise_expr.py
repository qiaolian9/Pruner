import tvm
from tvm import te
from tvm import auto_scheduler

@auto_scheduler.register_workload
def add_expr(shape, dataType='float32'):
    A = te.placeholder(shape, dtype=dataType, name="A")
    B = te.placeholder(shape, dtype=dataType, name="B")
    C = te.compute(shape, lambda *i: A(*i) + B(*i), name="C")
    return [A, B, C]

@auto_scheduler.register_workload
def relu_expr(shape, dataType='float32'):
    A = te.placeholder(shape, dtype=dataType, name="A")
    C = te.compute(shape, lambda *i: te.max(A(*i), tvm.tir.const(0, A.dtype)), name="C")
    return [A, C]

@auto_scheduler.register_workload
def relu_fusable_expr(shape, dataType='float32'):
    D1, D2, D3 = shape
    A = te.placeholder((D1, D2, D3), dtype=dataType, name="A")
    C = te.compute((D1, D2, D3), lambda d1, d2, d3: te.max(A[d1, d2, d3], tvm.tir.const(0, A.dtype)), name="C")
    return [A, C]