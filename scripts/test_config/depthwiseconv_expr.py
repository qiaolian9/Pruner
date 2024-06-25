from tvm import te, auto_scheduler

@auto_scheduler.register_workload
def depthwiseconv_expr_S2D1P0(shape, dataType="float32"):
    S, D, P = 2, 1, 0
    N, C, HO, WO, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    
    data = te.placeholder((N, C, H, W), dtype=dataType, name="data")
    kernel = te.placeholder((C, KH, KW), dtype=dataType, name="kernel")

    kh = te.reduce_axis((0, KH), name="kh")
    kw = te.reduce_axis((0, KW), name="kw")

    data_pad = te.compute((N, C, H + 2 * P, W + 2 * P),
                lambda n, c, ho, wo: te.if_then_else(te.all(P <= ho, ho < H + P, P <= wo, wo < W + P),
                data[n, c, ho, wo], 0.0),
                tag="data_pad", name="data_pad")

    depthwiseconv2d = te.compute(
        (N, C, HO, WO),
        lambda n, c, ho, wo: te.sum(
            (data_pad[n, c, ho * S + kh * D, wo * S + kw * D] * kernel[c, kh, kw]),
            axis=[kh, kw],
        ),
        name="depthwiseconv2d")

    return [data, kernel, depthwiseconv2d]


@auto_scheduler.register_workload
def depthwiseconv_expr_S1D1P1(shape, dataType="float32"):
    S, D, P = 1, 1, 1
    N, C, HO, WO, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    data = te.placeholder((N, C, H, W), dtype=dataType, name="data")
    kernel = te.placeholder((C, KH, KW), dtype=dataType, name="kernel")

    kh = te.reduce_axis((0, KH), name="kh")
    kw = te.reduce_axis((0, KW), name="kw")

    data_pad = te.compute((N, C, H + 2 * P, W + 2 * P),
                lambda n, c, ho, wo: te.if_then_else(te.all(P <= ho, ho < H + P, P <= wo, wo < W + P),
                data[n, c, ho, wo], 0.0),
                tag="data_pad", name="data_pad")

    depthwiseconv2d = te.compute(
        (N, C, HO, WO),
        lambda n, c, ho, wo: te.sum(
            (data_pad[n, c, ho * S + kh * D, wo * S + kw * D] * kernel[c, kh, kw]),
            axis=[kh, kw],
        ),
        name="depthwiseconv2d")

    return [data, kernel, depthwiseconv2d]

