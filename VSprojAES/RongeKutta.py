import numpy as np


def rk4(residexpl, mesh, bc, u, time, dt, nstep):
#RK4 Time integrator using a 4 stage Runge-Kutta scheme.
#   U=RK4(RESIDEXPL,MASTER,MESH,APP,U,TIME,DT,NSTEP)
#
#      RESIDEXPL:    Pointer to residual evaluation function
#                    R=RESIDEXPL(MASTER,MESH,APP,U,TIME)
#      MESH:         Mesh structure
#      bc:           boundary condition
#      U(NC,NT):     Vector of unknowns
#                    NC = app.nc (number of equations in system)
#                    NT = size(mesh.t,1)
#      TIME:         Time
#      DT:           Time step
#      NSTEP:        Number of steps to be performed
#      R(NPL,NC,NT): Residual vector (=dU/dt)

    for i in range(1, nstep):   
        k1 = dt*residexpl(mesh, u       , bc, time       )
        k2 = dt*residexpl(mesh, u+0.5*k1, bc, time+0.5*dt)
        k3 = dt*residexpl(mesh, u+0.5*k2, bc, time+0.5*dt)
        k4 = dt*residexpl(mesh, u+    k3, bc, time+    dt)
        u = u + k1/6 + k2/3 + k3/3 + k4/6
        time = time + dt

    return u


def rk1(residexpl, mesh, bc, u, time, dt, nstep):
    for i in range(1, nstep):
        k1 = dt* residexpl(mesh, u, bc, time)
        u = u + k1
        time = time + dt

    return u
