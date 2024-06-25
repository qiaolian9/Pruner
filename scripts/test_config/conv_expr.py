import os
from tvm import te, auto_scheduler

FUSE_PAD = (os.getenv('FUSE_PAD') == "1")
CNHW = (os.getenv('CONV_LAYOUT')=="CNHW")
NHWC = (os.getenv('DATA_LAYOUT')=="NHWC")


@auto_scheduler.register_workload
def conv_expr_S1D1P0(shape, dataType='float32'):
    S, D, P = 1, 1, 0
    N, F, HO, WO, C, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    data_shape = ((H, W) if FUSE_PAD else (H + 2 * P, W + 2 * P))
    data_shape = (N, *data_shape, C)
    data = te.placeholder(data_shape, dtype=dataType, name="data")
    kernel = te.placeholder((F, C, KH, KW), dtype=dataType, name="kernel")

    c = te.reduce_axis((0, C), name='c')
    kh = te.reduce_axis((0, KH), name='kh')
    kw = te.reduce_axis((0, KW), name='kw')

    conv = te.compute((N, F, HO, WO), lambda n, f, ho, wo:\
        te.sum(data[n, ho * S + kh * D, wo * S + kw * D, c] *
                kernel[f, c, kh, kw],
                axis=[c, kh, kw])
                , name='conv')


    return [data, kernel, conv]

@auto_scheduler.register_workload
def conv_expr_S1D1P0_NCHW(shape, dataType='float32'):
    S, D, P = 1, 1, 0
    N, F, HO, WO, C, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    data_shape = ((H, W) if FUSE_PAD else (H + 2 * P, W + 2 * P))
    data_shape = (N, C, *data_shape)
    data = te.placeholder(data_shape, dtype=dataType, name="data")
    kernel = te.placeholder((F, C, KH, KW), dtype=dataType, name="kernel")

    c = te.reduce_axis((0, C), name='c')
    kh = te.reduce_axis((0, KH), name='kh')
    kw = te.reduce_axis((0, KW), name='kw')

    conv = te.compute((N, F, HO, WO), lambda n, f, ho, wo:\
        te.sum(data[n, c, ho * S + kh * D, wo * S + kw * D] *
                kernel[f, c, kh, kw],
                axis=[c, kh, kw])
                , name='conv')


    return [data, kernel, conv]

@auto_scheduler.register_workload
def conv_expr_S1D1P1(shape, dataType="float32"):
    S, D, P = 1, 1, 1
    N, F, HO, WO, C, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P
   
    data_shape = ((H, W) if FUSE_PAD else (H + 2 * P, W + 2 * P))
    data_shape = (N, *data_shape, C)
    data = te.placeholder(data_shape, dtype=dataType, name="data")
    kernel = te.placeholder((F, C, KH, KW), dtype=dataType, name="kernel")

    c = te.reduce_axis((0, C), name='c')
    kh = te.reduce_axis((0, KH), name='kh')
    kw = te.reduce_axis((0, KW), name='kw')

    conv = te.compute((N, F, HO, WO), lambda n, f, ho, wo:\
        te.sum(data[n, ho * S + kh * D, wo * S + kw * D, c] *
                kernel[f, c, kh, kw],
                axis=[c, kh, kw])
                , name='conv')

    return [data, kernel, conv]

@auto_scheduler.register_workload
def conv_expr_S2D1P1(shape, dataType="float32"):
    S, D, P = 2, 1, 1
    N, F, HO, WO, C, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P + 1
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P + 1

    data_shape ((H, W) if FUSE_PAD else (H + 2 * P, W + 2 * P))
    data_shape = (N, *data_shape, C)
    data = te.placeholder(data_shape, dtype=dataType, name="data")
    kernel = te.placeholder((F, C, KH, KW), dtype=dataType, name="kernel")

    c = te.reduce_axis((0, C), name='c')
    kh = te.reduce_axis((0, KH), name='kh')
    kw = te.reduce_axis((0, KW), name='kw')

    conv = te.compute((N, F, HO, WO), lambda n, f, ho, wo:\
            te.sum(data[n, ho * S + kh * D, wo * S + kw * D, c] *
                    kernel[f, c, kh, kw],
                    axis=[c, kh, kw])
                    , name='conv')

    return [data, kernel, conv]

@auto_scheduler.register_workload
def conv_expr_S2D1P0(shape, dataType="float32"):
    S, D, P = 2, 1, 0
    N, F, HO, WO, C, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P + 1
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P + 1

    data_shape = ((H, W) if FUSE_PAD else (H + 2 * P, W + 2 * P))
    data_shape = (N, *data_shape, C) 
    data = te.placeholder(data_shape, dtype=dataType, name="data")
    kernel = te.placeholder((F, C, KH, KW), dtype=dataType, name="kernel")

    c = te.reduce_axis((0, C), name='c')
    kh = te.reduce_axis((0, KH), name='kh')
    kw = te.reduce_axis((0, KW), name='kw')

    conv = te.compute((N, F, HO, WO), lambda n, f, ho, wo:\
        te.sum(data[n, ho * S + kh * D, wo * S + kw * D, c] *
                kernel[f, c, kh, kw],
                axis=[c, kh, kw])
                , name='conv')

    return [data, kernel, conv]

@auto_scheduler.register_workload
def conv_expr_S2D1P0_NCHW(shape, dataType="float32"):
    S, D, P = 2, 1, 0
    N, F, HO, WO, C, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P + 1
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P + 1

    data_shape = ((H, W) if FUSE_PAD else (H + 2 * P, W + 2 * P))
    data_shape = (N, C, *data_shape)
    data = te.placeholder(data_shape, dtype=dataType, name="data")
    kernel = te.placeholder((F, C, KH, KW), dtype=dataType, name="kernel")

    c = te.reduce_axis((0, C), name='c')
    kh = te.reduce_axis((0, KH), name='kh')
    kw = te.reduce_axis((0, KW), name='kw')

    conv = te.compute((N, F, HO, WO), lambda n, f, ho, wo:\
        te.sum(data[n, c, ho * S + kh * D, wo * S + kw * D] *
                kernel[f, c, kh, kw],
                axis=[c, kh, kw])
                , name='conv')

    return [data, kernel, conv]

@auto_scheduler.register_workload
def fused_conv_expr_S2D1P0(shape, dataType="float32"):
    S, D, P = 2, 1, 0
    
    N, F, HO, WO, C, KH, KW = shape

    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P + 1
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P + 1

    sa_fused0_pad = N * HO * WO
    ra_fused0_pad = C * KH * KW
    f_pad = F
    
    data_shape = (C, N) if CNHW else (N,C)
    data_shape += ((H, W) if FUSE_PAD else (H + 2 * P, W + 2 * P))
    data = te.placeholder(data_shape, dtype=dataType, name="data")
    kernel = te.placeholder((F, C, KH, KW), dtype=dataType, name="kernel")

    ra_fused0 = te.reduce_axis((0, C * KH * KW), name="ra_fused0")

    data_pad = te.compute([ra_fused0_pad, sa_fused0_pad],
           lambda ra_fused0, sa_fused0: te.if_then_else(te.all(sa_fused0 < N * HO * WO, ra_fused0 < C * KH * KW),
           data[ra_fused0 // (KH * KW) if CNHW else sa_fused0 // (HO * WO),
                sa_fused0 // (HO * WO) if CNHW else ra_fused0 // (KH * KW),
                sa_fused0 % (HO * WO) // WO * S + ra_fused0 % (KH * KW) // (KW) * D,
                sa_fused0 % (HO * WO) % WO * S + ra_fused0 % (KH * KW) % (KW) * D], 0.0),
           tag="data_pad", name="data_pad")

    kernel_pad = te.compute([f_pad, ra_fused0_pad],
           lambda f, ra_fused0: te.if_then_else(te.all(f < F, ra_fused0 < C * KH * KW),
           kernel[f,
                ra_fused0 // (KH * KW),
                ra_fused0 % (KH * KW) // (KW),
                ra_fused0 % (KH * KW) % (KW)], 0.0),
           tag="kernel_pad", name="kernel_pad")


    conv = te.compute([f_pad, sa_fused0_pad], lambda f, sa_fused0:\
                te.sum(data_pad[ra_fused0, sa_fused0] *
                        kernel_pad[f, ra_fused0],
                        axis=[ra_fused0])
                        , name='conv')
    conv_unpad = te.compute([F, N * HO * WO], lambda f, sa_fused0: conv[f, sa_fused0], tag="conv_unpad", name="conv_unpad")

    return [data, kernel, data_pad, kernel_pad, conv, conv_unpad]



@auto_scheduler.register_workload
def fused_conv_expr_S1D1P0(shape, dataType="float32"):
    S, D, P = 1, 1, 0

    N, F, HO, WO, C, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    sa_fused0_pad = N * HO * WO
    ra_fused0_pad = C * KH * KW
    f_pad = F
    
    data_shape = (C, N) if CNHW else (N,C)
    data_shape += ((H, W) if FUSE_PAD else (H + 2 * P, W + 2 * P))
    data = te.placeholder(data_shape, dtype=dataType, name="data")
    kernel = te.placeholder((F, C, KH, KW), dtype=dataType, name="kernel")

    ra_fused0 = te.reduce_axis((0, C * KH * KW), name="ra_fused0")

    data_pad = te.compute([ra_fused0_pad, sa_fused0_pad],
           lambda ra_fused0, sa_fused0: te.if_then_else(te.all(sa_fused0 < N * HO * WO, ra_fused0 < C * KH * KW),
           data[ra_fused0 // (KH * KW) if CNHW else sa_fused0 // (HO * WO),
                sa_fused0 // (HO * WO) if CNHW else ra_fused0 // (KH * KW),
                sa_fused0 % (HO * WO) // WO * S + ra_fused0 % (KH * KW) // (KW) * D,
                sa_fused0 % (HO * WO) % WO * S + ra_fused0 % (KH * KW) % (KW) * D], 0.0),
           tag="data_pad", name="data_pad")

    kernel_pad = te.compute([f_pad, ra_fused0_pad],
           lambda f, ra_fused0: te.if_then_else(te.all(f < F, ra_fused0 < C * KH * KW),
           kernel[f,
                ra_fused0 // (KH * KW),
                ra_fused0 % (KH * KW) // (KW),
                ra_fused0 % (KH * KW) % (KW)], 0.0),
           tag="kernel_pad", name="kernel_pad")


    conv = te.compute([f_pad, sa_fused0_pad], lambda f, sa_fused0:\
                te.sum(data_pad[ra_fused0, sa_fused0] *
                        kernel_pad[f, ra_fused0],
                        axis=[ra_fused0])
                        , name='conv')
    conv_unpad = te.compute([F, N * HO * WO], lambda f, sa_fused0: conv[f, sa_fused0], tag="conv_unpad", name="conv_unpad")

    return [data, kernel, data_pad, kernel_pad, conv, conv_unpad]
