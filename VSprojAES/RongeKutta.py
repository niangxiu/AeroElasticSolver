import numpy as np
import math

def rk4(residexpl, mesh, prmt, u, time, dt, nstep):
#RK4 Time integrator using a 4 stage Runge-Kutta scheme.
#   U=RK4(RESIDEXPL,MASTER,MESH,APP,U,TIME,DT,NSTEP)
#
#      RESIDEXPL:    Pointer to residual evaluation function
#                    R=RESIDEXPL(MASTER,MESH,APP,U,TIME)
#      MESH:         Mesh structure
#      prmt:         some parameters
#      U(NC,NT):     Vector of unknowns
#                    NC = app.nc (number of equations in system)
#                    NT = size(mesh.t,1)
#      TIME:         Time
#      DT:           Time step
#      NSTEP:        Number of steps to be performed
#      R(NPL,NC,NT): Residual vector (=dU/dt)

    for i in range(0, nstep):   
        k1 = dt*residexpl(mesh, u       , prmt, time       )
        k2 = dt*residexpl(mesh, u+0.5*k1, prmt, time+0.5*dt)
        k3 = dt*residexpl(mesh, u+0.5*k2, prmt, time+0.5*dt)
        k4 = dt*residexpl(mesh, u+    k3, prmt, time+    dt)
        u = u + k1/6 + k2/3 + k3/3 + k4/6
        time = time + dt

    return u


def rk1FF(residexpl, mesh, prmt, u, time, dt, nstep):
    # Euler stepping forward for flow field
    for i in range(0, nstep):
        k1 = dt* residexpl(mesh, u, prmt, time)
        u = u + k1
        time = time + dt

    return u


def rk1S(residexpl, A, B, nmode, mesh, prmt, p, time, dt, nstep):
    # Euler stepping for structure
    # note dAdt = dAdtao * (tao/t), tao/t = (D/m/a^4)^0.5
    for i in range(0, nstep):
        [Ap1, Bp1] = residexpl( A, B, nmode, mesh, prmt,p )
        kA1 = Ap1 * math.sqrt(prmt.D / (prmt.m * prmt.a ** 4)) * dt
        dBdtao = Bp1 
        kB1 = Bp1 * math.sqrt(prmt.D / (prmt.m * prmt.a ** 4)) * dt
        A += kA1
        B += kB1
        time = time + dt
    return [A, B]
