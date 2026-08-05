"""Bivariate-bicycle code constructions.

`SPECS` holds each code's group orders and its two generator polynomials, and
`get_bb` turns one into the same plain dict the surface and colour constructions
return, so everything downstream treats all three families alike.

The one BB-specific detail is the qubit "coordinates": a BB code has no lattice
geometry, but its qubits split into a left and a right block, and that split is
what the checkerboard baselines need. Synthetic coordinates encode exactly that
split, which is why XZZX on a BB code means "Hadamard on one block".

Like multicode_deform, this module carried a standalone experiment driver that
the artifact never reached; it is removed rather than shipped.
"""
import numpy as np
from sympy.abc import x, y
# `qldpc` transitively imports cvxpy, which prints a solver-registration notice to
# stderr when the installed ortools is newer than the version cvxpy pins. The notice
# is harmless -- cvxpy is not used by Chameleon at all -- but it would appear on
# every command an evaluator runs and reads like an error. Silence just this import.
import contextlib as _ctx, io as _io
with _ctx.redirect_stderr(_io.StringIO()), _ctx.redirect_stdout(_io.StringIO()):
    from .._deps import required
    with required("qldpc", purpose="the bivariate-bicycle code construction"):
        from qldpc import codes

SPECS={"BB18":({x:3,y:3}, x**2+y+1, x+y**2+1),
       "BB30":({x:5,y:3}, 1+y+x*y**2, 1+x*y+x**3*y**2),
       "BB36":({x:3,y:6}, x+y**2+y**3, 1+y+x**2),        # [[36,4,6]] arXiv:2408.10001
       "BB72":({x:6,y:6}, x**3+y+y**2, y**3+x+x**2),        # [[72,12,6]]
       "BB90":({x:15,y:3}, x**9+y+y**2, 1+x**2+x**7),        # [[90,8,10]]
       "BB108":({x:9,y:6}, x**3+y+y**2, y**3+x+x**2),        # [[108,8,10]]
       "BB144":({x:12,y:6}, x**3+y+y**2, y**3+x+x**2)}       # [[144,12,12]]

def get_bb(name):
    """Build BB code ``name`` from SPECS as a plain dict (see the module docstring)."""
    o,A,B=SPECS[name]; c=codes.BBCode(o,A,B)
    Hx=np.array(c.matrix_x,dtype=np.uint8)%2; Hz=np.array(c.matrix_z,dtype=np.uint8)%2
    n=c.num_qubits; k=c.dimension; L=np.array(c.get_logical_ops(),dtype=np.uint8)%2
    LxX=L[:k,:n]; LzZ=L[k:2*k,n:]
    # left/right block 2-coloring for XZZX/ZXXZ-analog: coords give cb() a clean split
    coords=[(0,0) if i<n//2 else (0,2) for i in range(n)]
    return dict(Hx=Hx,Hz=Hz,LxX=LxX,LzZ=LzZ,coords=coords,n=n,k=k)


