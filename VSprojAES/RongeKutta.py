import numpy as np
import math
from Euler import Euler, get_p_H
from structure import plate, get_dydx_v_bot

# calculate the time derivative of both the fluid and the structure
def fs(mesh, prmt, u, AB, nmode, time):
    # set flow field boundary by structure
    A = AB[0]
    B  =AB[1]
    [mesh.w_bot, mesh.wx_bot, mesh.wt_bot] = get_dydx_v_bot(A, B, nmode, mesh, prmt)
    # solve flow field
    [p,temp] = get_p_H(u)
    dudt = Euler(mesh, u, prmt, time) 
    # solve structure
    dABdt = plate(AB, nmode, mesh, prmt, p)
    return dudt, dABdt



def rk4(mesh, prmt, u, AB, dt, nmode, time):
#RK4 Time integrator for the whole system
#   U=RK4(RESIDEXPL,MASTER,MESH,APP,U,TIME,DT,NSTEP)
#      MESH:         Mesh structure
#      prmt:         some parameters
#      U(NC,NT):     Vector of unknowns
#                    NC = app.nc (number of equations in system)
#                    NT = size(mesh.t,1)
#      DT:           Time step
    [du1, dAB1] = fs(mesh, prmt, u        , AB         , nmode, time)
    du1 = dt * du1
    dAB1 = dt * dAB1
    [du2, dAB2] = fs(mesh, prmt, u+0.5*du1, AB+0.5*dAB1, nmode, time)
    du2 = dt * du2
    dAB2 = dt * dAB2
    [du3, dAB3] = fs(mesh, prmt, u+0.5*du2, AB+0.5*dAB2, nmode, time)
    du3 = dt * du3
    dAB3 = dt * dAB3
    [du4, dAB4] = fs(mesh, prmt, u+    du3, AB+dAB3    , nmode, time)
    du4 = dt * du4
    dAB4 = dt * dAB4
    u  = u  + du1/6  + du2/3  + du3/3  + du4/6
    AB = AB + dAB1/6 + dAB2/3 + dAB3/3 + dAB4/6
    return u, AB



def rk1FF(residexpl, mesh, prmt, u, time, dt):
    # Euler stepping forward for flow field
   
    k1 = dt* residexpl(mesh, u, prmt, time)
    u = u + k1
    time = time + dt

    return u


def rk1S(residexpl, AB, nmode, mesh, prmt, p, time, dt):
    # Euler stepping for structure
    # note dAdt = dAdtao * (tao/t), tao/t = (D/m/a^4)^0.5
  
    dAB = dt * residexpl( AB, nmode, mesh, prmt,p )
    A += dAB[0]
    B += dAB[1]
    time = time + dt
    return [AB]
