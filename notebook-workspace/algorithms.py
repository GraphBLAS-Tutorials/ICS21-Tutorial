import pygraphblas as grb

#==========================================================================
def neighborhood(graph, src, num_hops):
    num_nodes = graph.nrows
    w = grb.Vector.sparse(grb.types.BOOL, num_nodes)
    v = grb.Vector.sparse(grb.types.BOOL, num_nodes)
    w[src] = True
    v.assign_scalar(True, mask=w)

    with grb.BOOL.LOR_LAND:
        for it in range(num_hops):
            w.vxm(graph, mask=v, out=w, desc=grb.descriptor.RC)
            v.assign_scalar(True, mask=w)

    return v

#==========================================================================
def pagerank(A, damping = 0.85, itermax = 100):
    n = A.nrows

    r = grb.Vector.sparse(grb.types.FP32, n)
    t = grb.Vector.sparse(grb.types.FP32, n)
    d = grb.Vector.sparse(grb.types.FP32, n)

    A.reduce_vector(out=d, mon=grb.types.FP32.PLUS_MONOID)

    d.assign_scalar(damping, accum=grb.types.FP32.DIV)
    r[:] = 1.0 / n
    teleport = (1 - damping) / n
    tol = 1e-4
    rdiff = 1.0
    for i in range(itermax):
        # swap t and r
        temp = t ; t = r ; r = temp
        w = t / d
        r[:] = teleport
        A.mxv(w,
              out=r,
              accum=grb.types.FP32.PLUS,
              semiring=grb.types.FP32.PLUS_SECOND,
              desc=grb.descriptor.T0)
        t -= r
        t.apply(grb.types.FP32.ABS, out=t)
        rdiff = t.reduce_float()
        if rdiff <= tol:
            break
    return r

#==========================================================================
def triangle_count(A):
    # Make sure A is symmetric, unweighted, with no self loops

    # Make unweighted
    A_unw = A.pattern(typ = grb.UINT64)

    # remove self loops (if any)
    A_nl = A_unw.offdiag()

    # make sure of symmetry
    A_nl.eadd(A_nl, add_op=grb.types.UINT64.LOR, desc=grb.descriptor.T1, out=A_nl)

    # count triangles
    C = A_nl.mxm(A_nl, semiring=grb.types.UINT64.PLUS_TIMES, mask=A_nl)
    count = C.reduce_int(grb.types.UINT64.PLUS_MONOID)
    return (int)(count/6.0)
