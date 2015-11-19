from math import sin, cos, pi, sqrt
from numpy import sum, zeros
import numpy as np

def plate(AB, nmode, mesh, prmt, p):
    # Ap, Bp: the derivative of A and B w.r.t. \tao, B = dAdtao
    # dAdt, dBdt: the derivative of A and B w.r.t. real time
    # note dAdt = dAdtao * (tao/t), tao/t = (D/m/a^4)^0.5
    A = AB[0]
    B = AB[1]
    Ap = B
    Bp = zeros(nmode + 1)
    for n in range(1, nmode + 1):
        v1 = 0.0
        v2 = 0.0
        v3 = 0.0
        v6 = 0.0
        v1 = A[n] * (n * pi) ** 4 / 2
        for r in range(1, nmode + 1):
            v2 += A[r] ** 2 * (r * pi) ** 2 / 2
        v2 = 6 * (1 - prmt.nu ** 2) * v2 * A[n] * (n * pi) ** 2 / 2
        v3 = prmt.Rx * A[n] * (n * pi) ** 2 / 2
        # v4 and v5 are piston theory
        v4 = 0.0
        v5 = 0.0
        for m in range(1, nmode + 1):
            if m != n:
                v4 += (float(n * m) / (n ** 2 - m ** 2)) * (1 - (-1) ** (m + n)) * A[m]
        v4 = prmt.lbd * v4
        v5 = prmt.lbd * sqrt(prmt.mu / (prmt.lbd * prmt.Mach)) * B[n] / 2
        # integration of aerodynamic force
        temp = p[mesh.x_num / 3 + 1: mesh.x_num * 2 / 3 + 1, 1] - prmt.p
        temp *= np.sin((n * pi * mesh.x_cell[mesh.x_num / 3 + 1: mesh.x_num * 2 / 3 + 1, 1]) / prmt.a)
        v6 = sum(temp) * mesh.dx * prmt.a ** 3 / (prmt.D * prmt.h)
        
        Bp[n] = -2.0 * (v1 + v2 + v3 + v6) # + v4 + v5)

    dAdt = Ap * np.sqrt(prmt.D / (prmt.m * prmt.a ** 4))
    dBdt = Bp * np.sqrt(prmt.D / (prmt.m * prmt.a ** 4))
    return np.array([dAdt, dBdt])

def get_w_wp(A, B, nmode):
    # calculate w and wp from galerkin modes
    w = 0.0
    wp = 0.0
    for n in range(1, nmode + 1):
        w += A[n] * sin(n * 0.75 * pi)
        wp += B[n] * sin(n * 0.75 * pi)
    return (w, wp)

def get_dydx_v_bot(A, B, nmode, mesh, prmt):
    # calculat the slope at the
    x = mesh.x_cell[:,1]
    y = zeros(mesh.x_num + 2)
    dydx = zeros(mesh.x_num + 2)
    v = zeros(mesh.x_num + 2)
    for n in range(1, nmode + 1):
        for i in range(mesh.x_num / 3 + 1, mesh.x_num * 2 / 3 + 1):
            y[i] += A[n] * sin(n * pi * x[i] / prmt.a) * prmt.h
            dydx[i] += A[n] * (n * pi / prmt.a) * cos(n * pi * x[i] / prmt.a) * prmt.h
            v[i] += B[n] * sin(n * pi * x[i] / prmt.a) * prmt.h * sqrt(prmt.D / (prmt.m * prmt.a ** 4))
    return (y, dydx, v)