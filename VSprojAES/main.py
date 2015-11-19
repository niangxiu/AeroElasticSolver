import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Euler import Euler, linEuler, get_p_H, get_Mach, get_Ptotal
from get_mesh import mesh_class
from parameter import parameter_class
from RongeKutta import rk1FF, rk1S, rk4
from outputClass import *
from structure import plate, get_w_wp, get_dydx_v_bot
from aec import DowellModel
import sys
import matplotlib.pyplot as plt


def main(end_time, lbd, nRx, fmod, smod):
    # fmod: the model of flow field: either Euler or linearize Euler
    # smod: structrual model
    Rx = -nRx * math.pi**2
    prmt = parameter_class('AE_dowell', end_time, Rx, lbd)
    # mesh
    x_num = 120
    y_num = 40
    mesh = mesh_class(x_num, y_num, prmt)
    # get initial gas states
    if fmod == 'Euler':
        u_cell = np.zeros((4, x_num + 2, y_num + 2))
        u_cell[0] = prmt.rhol
        u_cell[1] = prmt.rhol * prmt.ul
        u_cell[2] = prmt.rhol * prmt.vl
        u_cell[3] = prmt.rhoEl
        p_cell = np.zeros((x_num + 2, y_num + 2))
        [p_cell, temp] = get_p_H(u_cell)
    elif fmod == 'linEuler':
        # To use finite difference, consider only points at the center of the cell
        # here p is the purturbation 
        pq_cell = np.zeros((2, x_num + 2, y_num + 2))   
        u_cell = np.zeros((4, x_num + 2, y_num + 2)) # always zero, just for comply with other parts of the code
        p_cell = pq_cell[0] + prmt.p

    # structure unkowns
    nmode = 4
    A = np.zeros(nmode + 1)
    B = np.zeros(nmode + 1)
    A[1] = 0.1
    AB = np.zeros([2, nmode+1])
    AB[0] = A
    AB[1] = B
    # other preparation
    sigma = 1.0
    dt = sigma * min(mesh.dx, mesh.dy)
    ncycle = int(math.ceil(end_time / (dt)))
    dt = end_time / (ncycle)
    output = outputClass()
    w_all = []
    wp_all = []
    [w, wp] = get_w_wp(A, B, nmode)
    w_all = np.hstack((w_all, w))
    wp_all = np.hstack((wp_all, wp))
    [mesh.w_bot, mesh.wx_bot, mesh.wt_bot] = get_dydx_v_bot(A, B, nmode, mesh, prmt)
    ## open output file, write file title and initial states
    #output.writeEvery('linEuler',1, nRx, lbd, u_cell, p_cell, mesh, prmt, A, B, nmode, w, wp)
    ## output cycle
    #outinterval = ncycle / 100

    for i in range(0, ncycle):
        time = i * dt
        print(time)       
        # use RK-4 to step forward
        u_cell, AB =  rk4(mesh, prmt, u_cell, AB, dt, nmode, time)
        # do some post processing for output
        A = AB[0]
        B = AB[1]
        [p_cell, temp] = get_p_H(u_cell)
        [w, wp] = get_w_wp(A, B, nmode)
        w_all = np.hstack((w_all, w))
        wp_all = np.hstack((wp_all, wp))
        # write to file 
        output.writeEvery('Euler',2, nRx, lbd, u_cell, p_cell, mesh, prmt, A, B, nmode, w, wp)
    #    if i % outinterval ==  0:
    #        output.writeEvery('Euler', 1, nRx, lbd, u_cell, p_cell, mesh, prmt, A, B, nmode, w, wp)
    #output.writeEvery('Euler', 1, nRx, lbd, u_cell, p_cell, mesh, prmt, A, B, nmode, w, wp)
    
    # write result of Dowell's model
    fileDM = open(''.join(['lbd', str(int(lbd)), '-Rx', str(nRx), '-DowellModel.dat']), 'w')
    [wD_all, wpD_all] = DowellModel(fileDM, prmt)
    # save a phase plane figure of Euler model and Dowell model
    plt.rc('text', usetex=True)
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 16}
    plt.rc('font', **font)
    plt.clf()
    plt.plot(w_all, wp_all)
    plt.axis('auto')
    plt.xlabel(r"$W$")
    plt.ylabel(r"$W'$")
    plt.savefig(''.join(['lbd', str(int(lbd)), '-Rx', str(nRx), '-EulPhaPlot.png']))
    plt.clf()
    plt.plot(wD_all, wpD_all)
    plt.axis('auto')
    plt.xlabel(r"$W$")
    plt.ylabel(r"$W'$")
    plt.savefig(''.join(['lbd', str(int(lbd)), '-Rx', str(nRx), '-DowPhaPlot.png']))

    ## calculate some static characteristics
    #w_all = np.array(w_all)
    #wp_all = np.array(wp_all)
    #fileMoment = open(''.join(['lbd', str(int(lbd)), '-Rx', str(nRx), '-Moment.dat']), 'w')
    #fileMoment.write('{:f} {:f} \n'.format(w_all.sum()/w_all.size, wp_all.sum()/wp_all.size))
    #fileMoment.write('{:f} {:f} \n'.format(wD_all.sum()/wD_all.size, wpD_all.sum()/wpD_all.size))
    #fileMoment.write('{:f} {:f} \n'.format((w_all**2).sum()/w_all.size, (wp_all**2).sum()/wp_all.size))
    #fileMoment.write('{:f} {:f} \n'.format((wD_all**2).sum()/wD_all.size, (wpD_all**2).sum()/wpD_all.size))


end_time = 600.0
main(end_time, 150, 4, 'Euler', 'beam')

main(end_time, 300, 4, 'Euler', 'beam')
main(end_time, 200, 4, 'Euler', 'beam')
main(end_time, 115, 4, 'Euler', 'beam')
main(end_time, 50, 4, 'Euler', 'beam')

main(end_time, 150, 1, 'Euler', 'beam')
main(end_time, 150, 3, 'Euler', 'beam')
main(end_time, 150, 6, 'Euler', 'beam')

main(end_time, 75, 2, 'Euler', 'beam')
main(end_time, 225, 6, 'Euler', 'beam')
main(end_time, 300, 8, 'Euler', 'beam')
