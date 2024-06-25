from tvm import te, auto_scheduler


@auto_scheduler.register_workload
def avgpool2d_expr_S2P0(shape, dataType='float32'):
    S, D, P = 2, 1, 0
    B, C, HO, WO, KH, KW = shape
  
    HI = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    WI = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    kh = te.reduce_axis((0, KH), name="kh")
    kw = te.reduce_axis((0, KW), name="kw")
    
    data = te.placeholder((B, C, HI, WI), dtype=dataType, name="data")
    padded_data = te.compute((B, C, HI + 2 * P, WI + 2 * P),lambda b, c, ho, wo: 
                                te.if_then_else(
                                te.all(P <= ho, ho < P + HI, P <= wo, wo < P + WI), data[b, c, ho, wo], 0.0), 
            name="padded_data")

    avgpool2d = te.compute((B, C, HO, WO), lambda b, c, ho, wo: 
                te.sum(padded_data[b, c, ho * S + kh * D, wo * S + kw * D] / (KH * KW),
            axis=[kh, kw],
        ),
        name="avgpool2d")

    return [data, avgpool2d]

@auto_scheduler.register_workload
def avgpool2d_expr_S1P0(shape, dataType='float32'):
    S, D, P = 1, 1, 0
    B, C, HO, WO, KH, KW = shape
  
    HI = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    WI = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    kh = te.reduce_axis((0, KH), name="kh")
    kw = te.reduce_axis((0, KW), name="kw")
    
    data = te.placeholder((B, C, HI, WI), dtype=dataType, name="data")
    padded_data = te.compute((B, C, HI + 2 * P, WI + 2 * P),lambda b, c, ho, wo: 
                                te.if_then_else(
                                te.all(P <= ho, ho < P + HI, P <= wo, wo < P + WI), data[b, c, ho, wo], 0.0), 
    name="padded_data")

    avgpool2d = te.compute((B, C, HO, WO), lambda b, c, ho, wo: 
                te.sum(padded_data[b, c, ho * S + kh * D, wo * S + kw * D] / (KH * KW),
            axis=[kh, kw],
        ),
        name="avgpool2d")

    return [data, avgpool2d]


@auto_scheduler.register_workload
def maxpool2d_expr_S2P1(shape, dataType='float32'):
    S, D, P = 2, 1, 1
    B, C, HO, WO, KH, KW = shape
  
    HI = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    WI = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P


    kh = te.reduce_axis((0, KH), name="kh")
    kw = te.reduce_axis((0, KW), name="kw")
    
    data = te.placeholder((B, C, HI + 1, WI + 1), dtype=dataType, name="data")
    padded_data = te.compute((B, C, HI + 2 * P, WI + 2 * P),lambda b, c, ho, wo: 
                                te.if_then_else(
                                te.all(ho < P + HI, wo < P + WI), data[b, c, ho, wo], -3.402823e37), 
                                #te.all(P <= ho, ho < P + HI, P <= wo, wo < P + WI), data[b, ho, wo], 0.0), 
    name="padded_data")

    maxpool2d = te.compute((B, C, HO, WO), lambda b, c, ho, wo: 
                te.max(padded_data[b, c, ho * S + kh * D, wo * S + kw * D],
            axis=[kh, kw],
        ),
        name="maxpool2d")

    return [data, maxpool2d]
