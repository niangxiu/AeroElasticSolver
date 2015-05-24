from numpy import *
# import matplotlib.pyplot as plt
from IPython.core.debugger import Tracer
from get_mesh import get_mesh


def get_p_H(U):
    # get pressure(p) and enthalpy(H)
    r  = U[0,:] # r = roe, density
    ru = U[1,:]
    rv = U[2,:]
    re = U[3,:]

    gamma = 1.4
    p = (gamma - 1) * (re - 0.5 * ru**2/r -0.5 * rv**2/r)
    H = re / r + p / r
    return [p, H]


def calc_flux(U, p):
    r  = U[0,:] # r = roe, density
    ru = U[1,:]
    rv = U[2,:]
    re = U[3,:]

    f = array([ru, ru**2/r + p, ru*rv/r, (re+p)*ru/r])
    return f


def Roe_solver_2D_x(Ul, Ur):
    rl  = Ul[0,:] # r = roe, density
    rul = Ul[1,:]
    rvl = Ul[2,:]
    rel = Ul[3,:]
    rr  = Ur[0,:]
    rur = Ur[1,:]
    rvr = Ur[2,:]
    rer = Ur[3,:]

    gamma = 1.4
    ul = rul / rl
    ur = rur / rr
    vl = rvl / rl
    vr = rvr / rr
    [pl, Hl] = get_p_H(Ul)
    [pr, Hr] = get_p_H(Ur)

    # Roe average
    rhot = sqrt(rl * rr)
    ut = (sqrt(rl) * ul + sqrt(rr) * ur ) / (sqrt(rl) + sqrt(rr))
    vt = (sqrt(rl) * vl + sqrt(rr) * vr ) / (sqrt(rl) + sqrt(rr))
    Ht = (sqrt(rl) * Hl + sqrt(rr) * Hr ) / (sqrt(rl) + sqrt(rr))
    at2 = (gamma - 1) * (Ht - 0.5*(ut**2 + vt**2))
    at = sqrt(at2)

    # eigenvalue
    lbd1 = ut - at
    lbd2 = ut
    lbd3 = ut
    lbd4 = ut + at
    eps = 0.0001 * ones(shape(lbd1))
    lbd1 = maximum(abs(lbd1), eps)
    lbd2 = maximum(abs(lbd2), eps)
    lbd3 = maximum(abs(lbd3), eps)
    lbd4 = maximum(abs(lbd4), eps)

    # alpha
    drho = rr - rl # density: rho
    dp = pr - pl # pressure
    dv = vr - vl # velocity parallel to the interface
    du = ur - ul # velocity perpendicular to the interface

    alpha1 = (dp - rhot*at*du) / (2*at2)
    alpha2 = rhot * dv / at
    alpha3 = (at2*drho - dp) / at2
    alpha4 = (dp + rhot*at*du) / (2*at2)

    # r is the column in the right eigen matrix
    ee = ones(shape(rl))
    zz = zeros(shape(rl))
    r1 = array([ ee, ut-at, vt, Ht-at*ut            ])
    r2 = array([ zz, zz,    at, at*vt               ])
    r3 = array([ ee, ut,    vt, 0.5*(ut**2 + vt**2) ])
    r4 = array([ ee, ut+at, vt, Ht+at*ut            ])

    # get raldu = R|diag matrix|L * (Ur - Ul)
    raldu =   lbd1 * alpha1 * r1\
            + lbd2 * alpha2 * r2\
            + lbd3 * alpha3 * r3\
            + lbd4 * alpha4 * r4

    # get inviscous flux Fl and Fr
    Fl = calc_flux(Ul, pl)
    Fr = calc_flux(Ur, pr)

    # get flux, finally
    Flux = 0.5 * (Fl+Fr) - 0.5 * raldu

    # Tracer()()
    return Flux


def muscl_vanleer(u_left, u_center, u_right):
    eps = 0.000001
    d_right = u_right - u_center
    d_left = u_center - u_left
    B = (d_left * d_right) * (sign(d_left) + sign(d_right))\
        /(abs(d_left) + abs(d_right) + eps)
    u_center_left = u_center - 0.5 * B
    u_center_right = u_center + 0.5 * B
    return [u_center_left, u_center_right]




def Euler(mesh, u_initial, bc, time):

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
    u_cell_top = zeros(shape(u_cell)) # state variable on top interface in cell
    u_cell_bot = zeros(shape(u_cell))
    u_cell_lef = zeros(shape(u_cell))
    u_cell_rig = zeros(shape(u_cell))
    flux_x_interface = zeros([4, x_num+1, y_num]) # flux along x direction
    flux_y_interface = zeros([4, x_num, y_num+1]) # flux along y direction
    flux_y_interface_flip = zeros(shape(flux_y_interface)) # exchange velocity along x and y: [rho, rv, ru, rE]
    
    
    
    ## parameters for ghost cells
    # left boundary
    u_cell[0, 0, :] = bc.rhol
    u_cell[1, 0, :] = bc.rhol * bc.ul
    u_cell[2, 0, :] = bc.rhol * bc.vl
    u_cell[3, 0, :] = bc.rhoEl
    # right boundary
    u_cell[0, -1, :] = bc.rhor
    u_cell[1, -1, :] = bc.rhor * bc.ur
    u_cell[2, -1, :] = bc.rhor * bc.vr
    u_cell[3, -1, :] = bc.rhoEr
   #  ## top boundary for double Mach
   #  for i in range(0, x_num+2):
   #      if x_cell[i, -1] < x0 + (1+20*time)/sqrt(3):
   #          u_cell[0, i, -1] = bc.rhol
   #          u_cell[1, i, -1] = bc.rhol * bc.ul
   #          u_cell[2, i, -1] = bc.rhol * bc.vl
   #          u_cell[3, i, -1] = bc.rhoEl
   #      else:
   #          u_cell[0, i, -1] = bc.rhor
   #          u_cell[1, i, -1] = bc.rhor * bc.ur
   #          u_cell[2, i, -1] = bc.rhor * bc.vr
   #          u_cell[3, i, -1] = bc.rhoEr
    ## top boundary for bump problem
    u_cell[0, :, -1] = bc.rhol
    u_cell[1, :, -1] = bc.rhol * bc.ul
    u_cell[2, :, -1] = bc.rhol * bc.vl
    u_cell[3, :, -1] = bc.rhoEl
   #  # bottom BC for bump problem
    u_cell[0, :, 0] = u_cell[0, :, 1]
    u_cell[1, :, 0] = u_cell[1, :, 1]
    u_cell[2, :, 0] = -u_cell[2,:,1] + u_cell[1,:,1] * mesh.dydx_bot
    u_cell[3, :, 0] = u_cell[3, :, 1]


    # bottom BC for the double Mach problem
   #  # bottom boundary: the left part
   #  u_cell[0, :i0+1, 0] = bc.rhol
   #  u_cell[1, :i0+1, 0] = bc.rhol * bc.ul
   #  u_cell[2, :i0+1, 0] = bc.rhol * bc.vl
   #  u_cell[3, :i0+1, 0] = bc.rhoEl
   #  # bolltom boundary: right part, solid wall
   #  u_cell[0, i0+1:, 0] = u_cell[0, i0+1:, 1]
   #  u_cell[1, i0+1:, 0] = u_cell[1, i0+1:, 1]
   #  u_cell[2, i0+1:, 0] = -u_cell[2, i0+1:, 1]
   #  u_cell[3, i0+1:, 0] = u_cell[3, i0+1:, 1]

    ## boundary flux
    flux_x_interface[:,0,:] = \
        Roe_solver_2D_x(u_cell[:,0, 1:-1], u_cell[:,1, 1:-1])  # left
    flux_x_interface[:,-1,:]= \
        Roe_solver_2D_x(u_cell[:,-2,1:-1], u_cell[:,-1,1:-1]) # right

    # flip interchanges x and y axis
    u_cell_flip = \
        array([u_cell[0,:], u_cell[2,:], u_cell[1,:],u_cell[3,:]])
    # top
    flux_y_interface_flip[:,:,-1]= \
        Roe_solver_2D_x(u_cell_flip[:,1:-1,-2],u_cell_flip[:,1:-1,-1]) # top
   #  # bottom: no wall
   #  flux_y_interface_flip[:,:i0,0] = \
   #      Roe_solver_2D_x(u_cell_flip[:,1:i0+1,0], u_cell_flip[:,1:i0+1,1])  # bottom
    # bottom: solid wall
    [ptemp0, Htemp0] = get_p_H(u_cell_flip[:,1:-1,0])
    [ptemp1, Htemp1] = get_p_H(u_cell_flip[:,1:-1,1])
    u_solidwall = 0.5 * (u_cell_flip[:,1:-1,0] + u_cell_flip[:,1:-1,1])  # bottom
    p_solidwall = 0.5 * (ptemp1 + ptemp0)
    flux_y_interface_flip[:,:,0] = \
        calc_flux( u_solidwall, p_solidwall)

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

    u_cell_top_flip = array([\
                             u_cell_top[0,:], u_cell_top[2,:],\
                             u_cell_top[1,:], u_cell_top[3,:]])
    u_cell_bot_flip = array([\
                             u_cell_bot[0,:], u_cell_bot[2,:],\
                             u_cell_bot[1,:], u_cell_bot[3,:]])
    flux_y_interface_flip[:,:,1:-1] = Roe_solver_2D_x(\
        u_cell_top_flip[:,1:-1,1:-2], u_cell_bot_flip[:,1:-1,2:-1])

    flux_y_interface = array([\
            flux_y_interface_flip[0,:],flux_y_interface_flip[2,:],\
            flux_y_interface_flip[1,:],flux_y_interface_flip[3,:]])

#    dduu = \
#        + (flux_x_interface[:,:-1,:] - flux_x_interface[:,1:,:]) * dy * dt / dx\
#        + (flux_y_interface[:,:,:-1] - flux_y_interface[:,:,1:]) * dx * dt / dy
    
    u_cell_resi = zeros(u_cell.shape)
    u_cell_resi[:,1:-1,1:-1] = \
        + (flux_x_interface[:,:-1,:] - flux_x_interface[:,1:,:]) / dx\
        + (flux_y_interface[:,:,:-1] - flux_y_interface[:,:,1:]) / dy

    
    
    return u_cell_resi
