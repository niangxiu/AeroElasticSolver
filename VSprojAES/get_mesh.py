import numpy as np
import matplotlib.pyplot as plt
# from IPython.core.debugger import Tracer
class mesh_class:
    # used to generate the mesh class for double-Mach problem
    # data: x_num, y_num, dx, dy, x_cell, y_cell, x0, i0
    def __init__(self, x_num, y_num, prmt):
        # check if x_num is even, since the plate is on the first half
        if x_num % 3 != 0:
            print('error gm11:x _num should be 3n')
        # flag: 'dm' 'bp'
        [self.dx, self.dy, self.x_cell, self.y_cell] = get_mesh(x_num, y_num, prmt)
        self.x_num = x_num
        self.y_num = y_num
        if prmt.name == 'dm':
            self.x0 = 1.0 / 6.0
            for i in range(0, x_num + 2):
                if self.x_cell[i,0] < self.x0:
                    self.i0 = i  # i0 corresponds to x0
        elif prmt.name == 'bp': 
            # the slope on the bottom surface
            self.wx_bot = 0.05 * ((self.x_cell[:,0] > 1.0) & (self.x_cell[:,0] < 2.0)) * 2 * np.pi * np.sin(np.pi * (self.x_cell[:,0] - 1.0)) * np.cos(np.pi * (self.x_cell[:,0] - 1.0)) 
        elif prmt.name == 'Nibump':
            self.w_bot = np.zeros(x_num + 2)
            self.wx_bot = np.zeros(x_num + 2)
            self.wt_bot = np.zeros(x_num + 2) # the moving velocity of plate at the bottom
            self.wtt_bot = np.zeros(x_num +2)
            self.wtx_bot = np.zeros(x_num +2)
            self.wxx_bot = np.zeros(x_num +2)
            x = self.x_cell[:,0]
            r = 1.3 * prmt.a
            #r = prmt.a *(0.04**2 + 0.25) / 0.08
            #r = 400.0
            for i in range(0, x_num + 2):
                if((x[i] < prmt.a) & (x[i] >0)):
                    self.w_bot[i] = np.sqrt(r**2 -(1-x[i])**2) - 1.2*prmt.a
            for i in range(1, x_num + 1):
                #if((x[i] < prmt.a) & (x[i] >0)):
                    #self.wx_bot[i] = (1-x[i]) / np.sqrt(r**2 -(1-x[i])**2) 
                self.wx_bot[i] = (self.w_bot[i+1] - self.w_bot[i-1]) / (2*self.dx)
                self.wxx_bot[i] = (self.w_bot[i+1] -2*self.w_bot[i] + self.w_bot[i-1]) / (self.dx**2)
        elif prmt.name == 'AE_dowell':
            #self.wx_bot = 0.05 *
            #((self.x_cell[:,0]>1.0)&(self.x_cell[:,0]<2.0)) * 2 * np.pi *
            #np.sin(np.pi*(self.x_cell[:,0]-1.0)) *
            #np.cos(np.pi*(self.x_cell[:,0]-1.0))
            self.w_bot = np.zeros(x_num + 2)
            self.wx_bot = np.zeros(x_num + 2)
            self.wt_bot = np.zeros(x_num + 2) # the moving velocity of plate at the bottom

def get_cell_location(x_interface, y_interface):
    # [x_cell, y_cell] = get_cell_location(x_interface, y_interface)
    # get location of centers of cells, with interface coordinates as input
    x_cell = 0.5 * (x_interface[:-1] + x_interface[1:])
    x_cell = np.append(2 * x_cell[0] - x_cell[1], x_cell)
    x_cell = np.append(x_cell, 2 * x_cell[-1] - x_cell[-2])

    y_cell = 0.5 * (y_interface[:-1] + y_interface[1:])
    y_cell = np.append(2 * y_cell[0] - y_cell[1], y_cell)
    y_cell = np.append(y_cell, 2 * y_cell[-1] - y_cell[-2])

    x_cell, y_cell = np.meshgrid(x_cell, y_cell, indexing='ij')

    return [x_cell, y_cell]


def get_mesh(x_num, y_num, prmt):
    # [dx, dy, x_cell, y_cell] = get_mesh(x_num, y_num)
    # get the coordinates of centers of cells, include ghost cells
    x_left = - prmt.a
    x_right = 2 * prmt.a
    if prmt.name == 'Nibump':
        y_bottom = 0.0
        y_top = prmt.a
    else:
        y_bottom = 0.0
        y_top = 0.4

    dx = (x_right - x_left) / x_num
    dy = (y_top - y_bottom) / y_num
    x_interface = np.linspace(x_left, x_right, x_num + 1)
    y_interface = np.linspace(y_bottom, y_top, y_num + 1)
    [x_cell, y_cell] = get_cell_location(x_interface, y_interface)

    return [dx, dy, x_cell, y_cell]
