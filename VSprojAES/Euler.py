import numpy as np
# import matplotlib.pyplot as plt
# from IPython.core.debugger import Tracer
from get_mesh import get_mesh


def get_p_H(U):
    # get pressure(p) and enthalpy(H)
    r = U[0] # r = roe, density
    ru = U[1]
    rv = U[2]
    re = U[3]

    gamma = 1.4
    p = (gamma - 1) * (re - 0.5 * ru ** 2 / r - 0.5 * rv ** 2 / r)
    H = re / r + p / r
    return [p, H]

def get_Mach(U, p):
    gamma = 1.4
    rho = U[0]
    u = U[1] / U[0]
    v = U[2] / U[0]
    a = np.sqrt(gamma * p / rho)
    Mach = np.sqrt(u ** 2 + v ** 2) / a
    return Mach

def get_Ptotal(p, Mach, gam):
    ptotal = p *(1 + 0.5*(gam-1) * Mach**2)**(gam/(gam-1))
    return ptotal

def calc_flux(U, p):
    r = U[0] # r = roe, density
    ru = U[1]
    rv = U[2]
    re = U[3]

    f = np.array([ru, ru ** 2 / r + p, ru * rv / r, (re + p) * ru / r])
    return f


def Roe_solver_2D_x(Ul, Ur):
    rl = Ul[0] # r = roe, density
    rul = Ul[1]
    rvl = Ul[2]
    rel = Ul[3]
    rr = Ur[0]
    rur = Ur[1]
    rvr = Ur[2]
    rer = Ur[3]

    gamma = 1.4
    ul = rul / rl
    ur = rur / rr
    vl = rvl / rl
    vr = rvr / rr
    [pl, Hl] = get_p_H(Ul)
    [pr, Hr] = get_p_H(Ur)

    # Roe average
    rhot = np.sqrt(rl * rr)
    ut = (np.sqrt(rl) * ul + np.sqrt(rr) * ur) / (np.sqrt(rl) + np.sqrt(rr))
    vt = (np.sqrt(rl) * vl + np.sqrt(rr) * vr) / (np.sqrt(rl) + np.sqrt(rr))
    Ht = (np.sqrt(rl) * Hl + np.sqrt(rr) * Hr) / (np.sqrt(rl) + np.sqrt(rr))
    at2 = (gamma - 1) * (Ht - 0.5 * (ut ** 2 + vt ** 2))
    at = np.sqrt(at2)

    # eigenvalue
    lbd1 = ut - at
    lbd2 = ut
    lbd3 = ut
    lbd4 = ut + at
    eps = 0.000001 * np.ones(np.shape(lbd1))
    lbd1 = np.maximum(np.abs(lbd1), eps)
    lbd2 = np.maximum(np.abs(lbd2), eps)
    lbd3 = np.maximum(np.abs(lbd3), eps)
    lbd4 = np.maximum(np.abs(lbd4), eps)

    # alpha
    drho = rr - rl # density: rho
    dp = pr - pl # pressure
    dv = vr - vl # velocity parallel to the interface
    du = ur - ul # velocity perpendicular to the interface

    alpha1 = (dp - rhot * at * du) / (2 * at2)
    alpha2 = rhot * dv / at
    alpha3 = (at2 * drho - dp) / at2
    alpha4 = (dp + rhot * at * du) / (2 * at2)

    # r is the column in the right eigen matrix
    ee = np.ones(np.shape(rl))
    zz = np.zeros(np.shape(rl))
    r1 = np.array([ee, ut - at, vt, Ht - at * ut])
    r2 = np.array([zz, zz,    at, at * vt])
    r3 = np.array([ee, ut,    vt, 0.5 * (ut ** 2 + vt ** 2)])
    r4 = np.array([ee, ut + at, vt, Ht + at * ut])

    # get raldu = R|diag matrix|L * (Ur - Ul)
    raldu = lbd1 * alpha1 * r1\
            + lbd2 * alpha2 * r2\
            + lbd3 * alpha3 * r3\
            + lbd4 * alpha4 * r4

    # get inviscous flux Fl and Fr
    Fl = calc_flux(Ul, pl)
    Fr = calc_flux(Ur, pr)

    # get flux, finally
    Flux = 0.5 * (Fl + Fr) - 0.5 * raldu

    # Tracer()()
    return Flux


def muscl_vanleer(u_left, u_center, u_right):
    eps = 0.000001
    d_right = u_right - u_center
    d_left = u_center - u_left
    B = (d_left * d_right) * (np.sign(d_left) + np.sign(d_right))\
        / (np.abs(d_left) + np.abs(d_right) + eps)
    u_center_left = u_center - 0.5 * B
    u_center_right = u_center + 0.5 * B
    return [u_center_left, u_center_right]


def slipWall(U1):
    # slip wall boundary condition
    U0 = U1
    U0[2] = -U1[2]
    return U0

def bumpWall_totalPressure(U1, mesh, prmt):
    gam = prmt.gamma
    # this bump wall boundary condition keeps same total pressure as input parameter
    U0 = np.array(U1)
    # get p0
    [p1, temp] = get_p_H(U0) 
    p0 = p1
    # get M0
    M02 = 2/(gam-1) * ((prmt.ptotal/p0)**((gam-1)/gam) - 1)
    M0 = np.sqrt(M02)
    # get u0 and v0 by slip wall b.c.
    k = mesh.wx_bot
    vg = mesh.wt_bot
    u1 = U1[1] / U1[0]
    v1 = U1[2] / U1[0]
    T = k * u1 + 2 * vg - v1 
    u0 = ( np.sqrt((k ** 2 + 1) * (u1 ** 2 + v1 ** 2) - T ** 2) - k * T) / (k ** 2 + 1)
    v0 = k * u0 + T
    # get sound speed and rho
    vel0 = np.sqrt(u0**2 + v0**2)
    sound = vel0 / M0
    rho0 = gam * p0 / sound**2
    # get rhoE
    rhoE0 = p0/(gam -1) + 0.5*rho0*vel0**2
    # line up
    U0[0] = rho0
    U0[1] = U0[0] * u0
    U0[2] = U0[0] * v0
    U0[3] = rhoE0
    return U0

def bumpWall(U1, mesh, prmt):
    U0 = np.array(U1)
    k = mesh.wx_bot
    vg = mesh.wt_bot
    u1 = U1[1] / U1[0]
    v1 = U1[2] / U1[0]
    T = k * u1 + 2 * vg - v1 
    u0 = ( np.sqrt((k ** 2 + 1) * (u1 ** 2 + v1 ** 2) - T ** 2) - k * T) / (k ** 2 + 1)
    v0 = k * u0 + T
    U0[1] = U0[0] * u0
    U0[2] = U0[0] * v0
    #U0[2] = -U1[2] + 2 * (U1[1] * mesh.wx_bot + U1[0] * mesh.wt_bot)
    #U0[1] = sqrt(U1[1]**2 - U0[2]**2)
    return U0

def bumpWall_dpdn(U1, mesh, prmt):
    U0 = np.array(U1)
    p1 = np.zeros_like(U1[0])
    dpdx = np.zeros_like(U1[0])
    u1 = U1[1] / U1[0]
    v1 = U1[2] / U1[0]
    # get u0 and v0
    b1 = u1 + mesh.wx_bot*v1
    b2 = mesh.wx_bot*u1 + 2* mesh.wt_bot - v1
    u0 = ( b1 - b2 * mesh.wx_bot) / (mesh.wx_bot ** 2 + 1)
    v0 = ( b2 + b1 * mesh.wx_bot) / (mesh.wx_bot ** 2 + 1)
    uwall = (u0+u1)/2
    vwall = (v0+v1)/2
    # get p0
    [p1, temp] = get_p_H(U0) 
    temp = -U0[0] *(mesh.wtt_bot + 2* mesh.wtx_bot*uwall + mesh.wxx_bot*uwall**2) # almost dpdn
    dpdx[1:-1] = (p1[2:] - p1[:-2])/ (2*mesh.dx)
    dpdy = mesh.wx_bot * dpdx  + temp
    p0 = p1 - dpdy*mesh.dy
    # get rho0
    rho0 = U1[0] * (p0 / p1)**prmt.gamma
    # get rhoE0
    rhoE0 = p0/(prmt.gamma-1) + 0.5 * rho0*u0**2
    # line up
    U0[0] = rho0
    U0[1] = rho0 * u0
    U0[2] = rho0 * v0
    U0[3] = rhoE0
    return U0

def bumpWall_slip(U1, mesh, prmt):
    U0 = np.array(U1)
    p1 = np.zeros_like(U1[0])
    dpdx = np.zeros_like(U1[0])
    u1 = U1[1] / U1[0]
    v1 = U1[2] / U1[0]
    # get u0 and v0
    b1 = u1 + mesh.wx_bot*v1
    b2 = mesh.wx_bot*u1 + 2* mesh.wt_bot - v1
    u0 = ( b1 - b2 * mesh.wx_bot) / (mesh.wx_bot ** 2 + 1)
    v0 = ( b2 + b1 * mesh.wx_bot) / (mesh.wx_bot ** 2 + 1)
    # line up
    U0[1] = U0[0] * u0
    U0[2] = U0[0] * v0
    return U0

def superSonicIn(U1, prmt):
    U0 = U1
    U0[0] = prmt.rhol
    U0[1] = prmt.rhol * prmt.ul
    U0[2] = prmt.rhol * prmt.vl
    U0[3] = prmt.rhoEl 
    return U0

def subSonicIn(U1, prmt):
    # here M is determined by U1, the state inside the boundary
    # initialize
    gamma = 1.4
    dW1 = np.zeros(U1.shape[0])
    dW0 = np.zeros(U1.shape[0])
    W0 = np.zeros(U1.shape)
    U0 = np.zeros(U1.shape)
    for i in range(0, U1.shape[1]):
        # get W1
        [p1, H1] = get_p_H(U1[:,i])
        rho1 = U1[0,i]
        u1 = U1[1,i] / U1[0,i]
        v1 = U1[2,i] / U1[0,i]
        # get left eigenvector matrix
        M = np.array   ([[-gamma * p1 / rho1,  0,        0,  1],\
                    [0,   0,  np.sqrt(gamma * rho1 * p1),    0],\
                    [0,   np.sqrt(gamma * rho1 * p1),    0,  1],\
                    [0,   -np.sqrt(gamma * rho1 * p1),   0,  1]])
        # get dW1 = [drho, du, dv, dp]
        dW1[0] = rho1 - prmt.rho
        dW1[1] = u1 - prmt.u 
        dW1[2] = v1 - prmt.v
        dW1[3] = p1 - prmt.p
        # get right hand side
        rhs = np.dot(M, dW1)
        rhs[0] = 0
        rhs[1] = 0
        rhs[2] = 0
        # solve for dW0
        dW0 = np.linalg.solve( M, rhs)
        # solve for W0
        W0[0,i] = prmt.rho + dW0[0]
        W0[1,i] = prmt.u + dW0[1]
        W0[2,i] = prmt.v + dW0[2]
        W0[3,i] = prmt.p + dW0[3]
    # solve for U0
    U0[0] = W0[0]
    U0[1] = W0[0] * W0[1]
    U0[2] = W0[0] * W0[2]
    U0[3] = W0[3] / (gamma - 1) + 0.5 * W0[0] * (W0[1] ** 2 + W0[2] ** 2)
    return U0

def subSonicIn_M0(U1, prmt):
    # initialize
    gamma = 1.4
    dW1 = np.zeros(U1.shape)
    dW0 = np.zeros(U1.shape)
    W0 = np.zeros(U1.shape)
    U0 = np.zeros(U1.shape)
    rhs = np.zeros(U1.shape)
    # get left eigenvector matrix
    M = matrix([[-gamma * prmt.p / prmt.rho,  0,          0,  1],\
              [0,   0,  np.sqrt(gamma * prmt.rho * prmt.p),    0],\
              [0,   np.sqrt(gamma * prmt.rho * prmt.p),    0,  1],\
              [0,   -np.sqrt(gamma * prmt.rho * prmt.p),   0,  1]])
    # get dW1 = [drho, du, dv, dp]
    [p1, H1] = get_p_H(U1)
    dW1[0] = U1[0] - prmt.rho
    dW1[1] = U1[1] / U1[0] - prmt.u 
    dW1[2] = U1[2] / U1[0] - prmt.v
    dW1[3] = p1 - prmt.p
    # get right hand side
    rhs = M * dW1
    rhs[0] = 0
    rhs[1] = 0
    rhs[2] = 0
    # solve for dW0
    dW0 = array(M ** -1 * rhs)
    # solve for W0
    W0[0] = prmt.rho + dW0[0]
    W0[1] = prmt.u + dW0[1]
    W0[2] = prmt.v + dW0[2]
    W0[3] = prmt.p + dW0[3]
    # solve for U0
    U0[0] = W0[0]
    U0[1] = W0[0] * W0[1]
    U0[2] = W0[0] * W0[2]
    U0[3] = W0[3] / (gamma - 1) + 0.5 * W0[0] * (W0[1] ** 2 + W0[2] ** 2)
    return U0

def superSonicOut(U1, prmt):
    U0 = array(U1)
    return U0

def subSonicOut(U1, prmt):
    # here M is determined by U1, the state inside the boundary
    # initialize
    gamma = 1.4
    dW1 = np.zeros(U1.shape[0])
    dW0 = np.zeros(U1.shape[0])
    W0 = np.zeros(U1.shape)
    U0 = np.zeros(U1.shape)
    for i in range(0, U1.shape[1]):
        # get W1
        [p1, H1] = get_p_H(U1[:,i])
        rho1 = U1[0,i]
        u1 = U1[1,i] / U1[0,i]
        v1 = U1[2,i] / U1[0,i]
        # get left eigenvector matrix
        M = np.array([[-gamma * p1 / rho1,  0,        0,  1],\
                    [0,   0,  np.sqrt(gamma * rho1 * p1),    0],\
                    [0,   np.sqrt(gamma * rho1 * p1),    0,  1],\
                    [0,   -np.sqrt(gamma * rho1 * p1),   0,  1]])
        # get dW1 = [drho, du, dv, dp]
        dW1[0] = rho1 - prmt.rho
        dW1[1] = u1 - prmt.u 
        dW1[2] = v1 - prmt.v
        dW1[3] = p1 - prmt.p
        # get right hand side
        rhs = np.dot(M, dW1)
        rhs[3] = 0
        # solve for dW0
        dW0 = np.linalg.solve( M, rhs)
        # solve for W0
        W0[0,i] = prmt.rho + dW0[0]
        W0[1,i] = prmt.u + dW0[1]
        W0[2,i] = prmt.v + dW0[2]
        W0[3,i] = prmt.p + dW0[3]
    # solve for U0
    U0[0] = W0[0]
    U0[1] = W0[0] * W0[1]
    U0[2] = W0[0] * W0[2]
    U0[3] = W0[3] / (gamma - 1) + 0.5 * W0[0] * (W0[1] ** 2 + W0[2] ** 2)
    return U0

def subSonicOut_M0(U1, prmt):
    # use intial flow state, subscript 0, to get M
    # initialize
    gamma = 1.4
    dW1 = np.zeros(U1.shape)
    dW0 = np.zeros(U1.shape)
    W0 = np.zeros(U1.shape)
    U0 = np.zeros(U1.shape)
    rhs = np.zeros(U1.shape)
    # get left eigenvector matrix
    M = np.matrix([[-gamma * prmt.p / prmt.rho,  0,          0,  1],\
              [0,   0,  np.sqrt(gamma * prmt.rho * prmt.p),    0],\
              [0,   np.sqrt(gamma * prmt.rho * prmt.p),    0,  1],\
              [0,   -np.sqrt(gamma * prmt.rho * prmt.p),   0,  1]])
    # get dW1 = [drho, du, dv, dp]
    [p1, H1] = get_p_H(U1)
    dW1[0] = U1[0] - prmt.rho
    dW1[1] = U1[1] / U1[0] - prmt.u 
    dW1[2] = U1[2] / U1[0] - prmt.v
    dW1[3] = p1 - prmt.p
    # get right hand side
    rhs = M * dW1
    rhs[3] = 0
    # solve for dW0
    dW0 = np.array(M ** -1 * rhs)
    # solve for W0
    W0[0] = prmt.rho + dW0[0]
    W0[1] = prmt.u + dW0[1]
    W0[2] = prmt.v + dW0[2]
    W0[3] = prmt.p + dW0[3]
    # solve for U0
    U0[0] = W0[0]
    U0[1] = W0[0] * W0[1]
    U0[2] = W0[0] * W0[2]
    U0[3] = W0[3] / (gamma - 1) + 0.5 * W0[0] * (W0[1] ** 2 + W0[2] ** 2)
    return U0


def doubleMachUpper():
    # todo: this function is not working
       #  ## top boundary for double Mach
    for i in range(0, x_num + 2):
        if x_cell[i, -1] < x0 + (1 + 20 * time) / sqrt(3):
            u_cell[0, i, -1] = prmt.rhol
            u_cell[1, i, -1] = prmt.rhol * prmt.ul
            u_cell[2, i, -1] = prmt.rhol * prmt.vl
            u_cell[3, i, -1] = prmt.rhoEl
        else:
            u_cell[0, i, -1] = prmt.rhor
            u_cell[1, i, -1] = prmt.rhor * prmt.ur
            u_cell[2, i, -1] = prmt.rhor * prmt.vr
            u_cell[3, i, -1] = prmt.rhoEr
    return U0
    pass

def doubleMachLower():
    # todo: this function is not working
    # bottom prmt for the double Mach problem
    # bottom boundary: the left part
    u_cell[0, :i0 + 1, 0] = prmt.rhol
    u_cell[1, :i0 + 1, 0] = prmt.rhol * prmt.ul
    u_cell[2, :i0 + 1, 0] = prmt.rhol * prmt.vl
    u_cell[3, :i0 + 1, 0] = prmt.rhoEl
    # bolltom boundary: right part, solid wall
    u_cell[0, i0 + 1:, 0] = u_cell[0, i0 + 1:, 1]
    u_cell[1, i0 + 1:, 0] = u_cell[1, i0 + 1:, 1]
    u_cell[2, i0 + 1:, 0] = -u_cell[2, i0 + 1:, 1]
    u_cell[3, i0 + 1:, 0] = u_cell[3, i0 + 1:, 1]
    return U0
    pass

def exactBotFlux():
    # exact zero boundary flux, do not use Roe directly because the entropy fix
    # may make it non-zero
    flux_y_interface_flip[:,:i0,0] = \
        Roe_solver_2D_x(u_cell_flip[:,1:i0 + 1,0], u_cell_flip[:,1:i0 + 1,1])  # bottom
    # bottom: solid wall
    [ptemp0, Htemp0] = get_p_H(u_cell_flip[:,1:-1,0])
    [ptemp1, Htemp1] = get_p_H(u_cell_flip[:,1:-1,1])
    u_solidwall = 0.5 * (u_cell_flip[:,1:-1,0] + u_cell_flip[:,1:-1,1])  # bottom
    p_solidwall = 0.5 * (ptemp1 + ptemp0)
    flux_y_interface_flip[:,:,0] = \
        calc_flux(u_solidwall, p_solidwall)


def Euler(mesh, u_initial, prmt, time):

    # preparation
    x_num = mesh.x_num
    y_num = mesh.y_num
    dx = mesh.dx
    dy = mesh.dy
    x_cell = mesh.x_cell
    y_cell = mesh.y_cell
    #x0 = mesh.x0
    #i0 = mesh.i0

    
    u_cell = u_initial
    # other matrix allocation
    u_cell_top = np.zeros(u_cell.shape) # state variable on top interface in cell
    u_cell_bot = np.zeros(u_cell.shape)
    u_cell_lef = np.zeros(u_cell.shape)
    u_cell_rig = np.zeros(u_cell.shape)
    flux_x_interface = np.zeros([4, x_num + 1, y_num]) # flux along x direction
    flux_y_interface = np.zeros([4, x_num, y_num + 1]) # flux along y direction
    flux_y_interface_flip = np.zeros(flux_y_interface.shape) # exchange velocity along x and y: [rho, rv, ru, rE]
    
    
    #u_cell[:, 0, :] = superSonicIn(u_cell[:, 1, :], prmt)# left boundary, super-sonic inlet
    u_cell[:, 0, :] = subSonicIn(u_cell[:, 1, :], prmt)# left boundary, sub-sonic inlet
    #u_cell[:, 0, :] = u_cell[:, 1, :]

    #u_cell[:, -1, :] = superSonicOut(u_cell[:, -2, :], prmt)# right boundary, supersonic outlet
    u_cell[:, -1, :] = subSonicOut(u_cell[:, -2, :], prmt)# right boundary, sub-sonic outlet
    #u_cell[:, -1, :] = u_cell[:, -2, :]

    # lower boundary, bump + slip wall
    u_cell[:, :, 0] = bumpWall(u_cell[:, :, 1], mesh, prmt)

    # upper boundary
    #u_cell[:, :, -1] = slipWall(u_cell[:, :, -2]) # slip wall
    u_cell[:, :, -1] = superSonicIn(u_cell[:, :, -2], prmt) # supersonic inlet

    ## boundary flux
    flux_x_interface[:,0,:] = \
        Roe_solver_2D_x(u_cell[:,0, 1:-1], u_cell[:,1, 1:-1])  # left
    flux_x_interface[:,-1,:] = \
        Roe_solver_2D_x(u_cell[:,-2,1:-1], u_cell[:,-1,1:-1]) # right

    # flip interchanges x and y axis
    u_cell_flip = \
        np.array([u_cell[0,:], u_cell[2,:], u_cell[1,:],u_cell[3,:]])
    # top
    flux_y_interface_flip[:,:,-1] = \
        Roe_solver_2D_x(u_cell_flip[:,1:-1,-2],u_cell_flip[:,1:-1,-1]) # top
    # bottom
    flux_y_interface_flip[:,:,0] = \
        Roe_solver_2D_x(u_cell_flip[:,1:-1,0], u_cell_flip[:,1:-1,1])  # bottom
   


# reconstruction for inner cells
    [u_cell_lef[:,1:-1,1:-1], u_cell_rig[:,1:-1,1:-1]] = \
        muscl_vanleer(u_cell[:, :-2, 1:-1], u_cell[:, 1:-1, 1:-1], \
                      u_cell[:, 2:, 1:-1])
    [u_cell_bot[:,1:-1,1:-1], u_cell_top[:,1:-1,1:-1]] = \
        muscl_vanleer(u_cell[:, 1:-1, :-2], u_cell[:, 1:-1, 1:-1], \
                      u_cell[:, 1:-1, 2:])

# inner field flux
    flux_x_interface[:,1:-1,:] = Roe_solver_2D_x(\
        u_cell_rig[:,1:-2,1:-1], u_cell_lef[:,2:-1,1:-1])

    u_cell_top_flip = np.array([\
                             u_cell_top[0,:], u_cell_top[2,:],\
                             u_cell_top[1,:], u_cell_top[3,:]])
    u_cell_bot_flip = np.array([\
                             u_cell_bot[0,:], u_cell_bot[2,:],\
                             u_cell_bot[1,:], u_cell_bot[3,:]])
    flux_y_interface_flip[:,:,1:-1] = Roe_solver_2D_x(\
        u_cell_top_flip[:,1:-1,1:-2], u_cell_bot_flip[:,1:-1,2:-1])

    flux_y_interface = np.array([\
            flux_y_interface_flip[0,:],flux_y_interface_flip[2,:],\
            flux_y_interface_flip[1,:],flux_y_interface_flip[3,:]])
        
    u_cell_resi = np.zeros(u_cell.shape)
    u_cell_resi[:,1:-1,1:-1] = \
        + (flux_x_interface[:,:-1,:] - flux_x_interface[:,1:,:]) / dx\
        + (flux_y_interface[:,:,:-1] - flux_y_interface[:,:,1:]) / dy

    
    
    return u_cell_resi
