import pygraphblas as grb

#==========================================================================
def neighborhood(graph, src, num_hops):
    num_nodes = graph.nrows
    w = grb.Vector.sparse(grb.BOOL, num_nodes)
    v = grb.Vector.sparse(grb.BOOL, num_nodes)
    w[src] = True
    v.assign_scalar(True, mask=w)

    with grb.BOOL.LOR_LAND:
        for it in range(num_hops):
            w.vxm(graph, mask=v, out=w, desc=grb.descriptor.RC)
            v.assign_scalar(True, mask=w)

    return v

#==========================================================================
def pagerank(A, damping = 0.85, tol = 1e-4, itermax = 100):
    n = A.nrows

    r = grb.Vector.dense(grb.FP32, n, fill=1.0/n)
    t = grb.Vector.dense(grb.FP32, n)
    d = grb.Vector.dense(grb.FP32, n, fill=damping)

    A.reduce_vector(out=d, accum=grb.FP32.DIV,
                    mon=grb.FP32.PLUS_MONOID)

    teleport = (1 - damping) / n

    for i in range(itermax):
        t[:] = r[:]
        w = t * d
        r[:] = teleport
        A.mxv(w,
              out=r,
              accum=grb.FP32.PLUS,
              semiring=grb.FP32.PLUS_SECOND,
              desc=grb.descriptor.T0)
        t -= r
        t.apply(grb.FP32.ABS, out=t)
        rdiff = t.reduce_float(grb.FP32.PLUS_MONOID)
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
    A_nl.eadd(A_nl, add_op=grb.UINT64.LOR, desc=grb.descriptor.T1, out=A_nl)

    # count triangles
    C = A_nl.mxm(A_nl, semiring=grb.UINT64.PLUS_TIMES, mask=A_nl)
    count = C.reduce_int(grb.UINT64.PLUS_MONOID)
    return (int)(count/6.0)
