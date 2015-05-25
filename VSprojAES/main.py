import numpy as np
import math
# from IPython.core.debugger import Tracer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Euler import Euler
from get_mesh import mesh_class
from boundary import boundary_class
from RongeKutta import rk1

bc = boundary_class('bp')
x_num = 90
y_num = 30
mesh = mesh_class(x_num, y_num, 'bp')
u_cell = np.zeros((4, x_num+2, y_num+2))

## the boundary state for the double MAch
# # initial gas state in flow field
# for i in range(0, x_num+2):
#     for j in range(0, y_num+2):
#         if mesh.x_cell[i,j] < mesh.x0 + mesh.y_cell[i,j] / np.sqrt(3):
#             u_cell[0, i, j] = bc.rhol
#             u_cell[1, i, j] = bc.rhol * bc.ul
#             u_cell[2, i, j] = bc.rhol * bc.vl
#             u_cell[3, i, j] = bc.rhoEl
#         else:
#             u_cell[0, i, j] = bc.rhor
#             u_cell[1, i, j] = bc.rhor * bc.ur
#             u_cell[2, i, j] = bc.rhor * bc.vr
#             u_cell[3, i, j] = bc.rhoEr
u_cell[0] = bc.rhol
u_cell[1] = bc.rhol * bc.ul
u_cell[2] = bc.rhol * bc.vl
u_cell[3] = bc.rhoEl


end_time = 1.5
sigma = 0.01
dt = 0.001 #sigma * min(mesh.dx, mesh.dy)
nstep = 10 
ncycle = int(math.ceil(end_time / (nstep * dt)))
dt = end_time / (nstep * ncycle)

f = open('result.dat', 'w')
f.write('title = "contour"\n')
f.write('variables = "x", "y", "rho", "rhou", "rhov", "rhoE" \n')
f.write('\n')
f.write('zone i = {:d}   j = {:d}   f = point\n'.format(y_num, x_num))
for i in range(1, x_num+1):
    for j in range(1, y_num+1):
        f. write('{:f} {:f} {:f} {:f} {:f} {:f}\n'.format(mesh.x_cell[i,j], mesh.y_cell[i,j],\
            u_cell[0,i,j], u_cell[1,i,j], u_cell[2,i,j], u_cell[3,i,j]) )

for i in range(0, ncycle):
    time = i * nstep * dt
    print(time) 
    u_cell = rk1(Euler, mesh, bc, u_cell, time, dt, nstep)
    f.write('\n')
    f.write('zone i = {:d}   j = {:d}   f = point\n'.format(y_num, x_num))
    for i in range(1, x_num+1):
        for j in range(1, y_num+1):
            f. write('{:f} {:f} {:f} {:f} {:f} {:f}\n'.format(mesh.x_cell[i,j], mesh.y_cell[i,j],\
                u_cell[0,i,j], u_cell[1,i,j], u_cell[2,i,j], u_cell[3,i,j]) )
    

#im = plt.contourf(mesh.x_cell[1:-1,1:-1], mesh.y_cell[1:-1,1:-1], u_cell[0,1:-1,1:-1], 30)
#plt.colorbar(im)
#plt.show()

