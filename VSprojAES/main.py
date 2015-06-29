import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Euler import Euler, get_p_H, get_Mach, get_Ptotal
from get_mesh import mesh_class
from parameter import parameter_class
from RongeKutta import rk1FF, rk1S
from outputClass import *
from structure import plate, get_w_wp, get_dydx_v_bot
from aec import DowellModel


def main(end_time, lbd, nRx):
    #end_time = 5.0
    #lbd = 200.0
    #nRx = 4
    Rx = -nRx * math.pi**2
    prmt = parameter_class('AE_dowell', end_time, Rx, lbd)
    # mesh
    x_num = 90
    y_num = 30
    mesh = mesh_class(x_num, y_num, prmt)
    # get initial gas states
    u_cell = np.zeros((4, x_num + 2, y_num + 2))
    u_cell[0] = prmt.rhol
    u_cell[1] = prmt.rhol * prmt.ul
    u_cell[2] = prmt.rhol * prmt.vl
    u_cell[3] = prmt.rhoEl
    p_cell = np.zeros((x_num + 2, y_num + 2))
    [p_cell, temp] = get_p_H(u_cell)
    # structure unkowns
    nmode = 4
    A = np.zeros(nmode + 1)
    B = np.zeros(nmode + 1)
    dBdtao = np.zeros(nmode + 1)
    Anew = np.zeros(nmode + 1)
    Bnew = np.zeros(nmode + 1)
    dBdtaonew = np.zeros(nmode + 1)
    A[1] = 0.1
    # other preparation
    sigma = 0.01
    dt = 0.001#sigma * min(mesh.dx, mesh.dy)
    nstep = 1 
    ncycle = int(math.ceil(end_time / (nstep * dt)))
    dt = end_time / (nstep * ncycle)
    output = outputClass()
    w_all = []
    wp_all = []
    [w, wp] = get_w_wp(A, B, nmode)
    w_all = np.hstack((w_all, w))
    wp_all = np.hstack((wp_all, wp))
    #[mesh.w_bot, mesh.wx_bot, mesh.wt_bot] = get_dydx_v_bot(A, B, nmode, mesh, prmt)
    # open output file, write file title and initial states
    output.writeEvery(1, nRx, lbd, u_cell, p_cell, mesh, prmt, A, B, nmode, w, wp)
    # output cycle
    outinterval = ncycle / 100

    for i in range(0, ncycle):
        time = i * nstep * dt
        print(time) 
        # set flow field boundary by structure
        #[mesh.w_bot, mesh.wx_bot, mesh.wt_bot] = get_dydx_v_bot(A, B, nmode, mesh, prmt)
        # solve flow field
        u_new = rk1FF(Euler, mesh, prmt, u_cell, time, dt, nstep)
        [p_new,temp] = get_p_H(u_new)
        # solve structure
        #[A_new, B_new] = rk1S(plate, A, B, nmode, mesh, prmt, p_new, time, dt, nstep)
        # step
        u_cell = u_new
        p_cell = p_new
        #A = A_new
        #B = B_new
        # get w and wp
        #[w, wp] = get_w_wp(A, B, nmode)
        #w_all = np.hstack((w_all, w))
        #wp_all = np.hstack((wp_all, wp))
        # write to file 
        output.writeEvery(2, nRx, lbd, u_cell, p_cell, mesh, prmt, A, B, nmode, w, wp)
        if i % outinterval ==  0:
            output.writeEvery(3, nRx, lbd, u_cell, p_cell, mesh, prmt, A, B, nmode, w, wp)
           
    output.writeEvery(1, nRx, lbd, u_cell, p_cell, mesh, prmt, A, B, nmode, w, wp)
    fileDM = open(''.join(['lbd', str(int(lbd)), '-Rx', str(nRx), '-DowellModel.dat']), 'w')
    [wD_all, wpD_all] = DowellModel(fileDM, prmt)
    filePiston = open(''.join(['lbd', str(int(lbd)), '-Rx', str(nRx), '-PistonTheory.dat']), 'w')
    output.writePiston(filePiston, prmt, mesh)

    w_all = np.array(w_all)
    wp_all = np.array(wp_all)
    fileMoment = open(''.join(['lbd', str(int(lbd)), '-Rx', str(nRx), '-Moment.dat']), 'w')
    fileMoment.write('{:f} {:f} \n'.format(w_all.sum()/w_all.size, wp_all.sum()/wp_all.size))
    fileMoment.write('{:f} {:f} \n'.format(wD_all.sum()/wD_all.size, wpD_all.sum()/wpD_all.size))
    fileMoment.write('{:f} {:f} \n'.format((w_all**2).sum()/w_all.size, (wp_all**2).sum()/wp_all.size))
    fileMoment.write('{:f} {:f} \n'.format((wD_all**2).sum()/wD_all.size, (wpD_all**2).sum()/wpD_all.size))



end_time = 100.0
main(end_time, 300, 4)
#main(end_time, 200, 4)
#main(end_time, 150, 4)
#main(end_time, 115, 4)
#main(end_time, 50, 4)

#main(end_time, 150, 1)
#main(end_time, 150, 3)
#main(end_time, 150, 6)