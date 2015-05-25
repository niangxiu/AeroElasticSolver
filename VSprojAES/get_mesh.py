import numpy as np
import matplotlib.pyplot as plt
# from IPython.core.debugger import Tracer

class mesh_class:
    # used to generate the mesh class for double-Mach problem
    # data: x_num, y_num, dx, dy, x_cell, y_cell, x0, i0
    def __init__(self, x_num, y_num, flag):
        # flag: 'dm' 'bp'
        [self.dx, self.dy, self.x_cell, self.y_cell] = get_mesh(x_num, y_num)
        self.x_num = x_num
        self.y_num = y_num
        if flag == 'dm':
            self.x0 = 1.0/ 6.0
            for i in range(0, x_num+2):
                if self.x_cell[i,0] < self.x0:
                    self.i0 = i  # i0 corresponds to x0
        elif flag == 'bp': 
            # the slope on the bottom surface
            self.dydx_bot = 0.05 * ((self.x_cell[:,0]>1.0)&(self.x_cell[:,0]<2.0)) * 2 * np.pi * np.sin(np.pi*(self.x_cell[:,0]-1.0)) * np.cos(np.pi*(self.x_cell[:,0]-1.0)) 

def get_cell_location(x_interface, y_interface):
    # [x_cell, y_cell] = get_cell_location(x_interface, y_interface)
    # get location of centers of cells, with interface coordinates as input
    x_cell = 0.5 * (x_interface[:-1] + x_interface[1:])
    x_cell = np.append(2 * x_cell[0] - x_cell[1], x_cell)
    x_cell = np.append(x_cell, 2*x_cell[-1] - x_cell[-2])

    y_cell = 0.5 * (y_interface[:-1] + y_interface[1:])
    y_cell = np.append(2 * y_cell[0] - y_cell[1], y_cell)
    y_cell = np.append(y_cell, 2 * y_cell[-1] - y_cell[-2])

    x_cell, y_cell = np.meshgrid(x_cell, y_cell, indexing='ij')

    return [x_cell, y_cell]


def get_mesh(x_num, y_num):
    # [dx, dy, x_cell, y_cell] = get_mesh(x_num, y_num)
    # get the coordinates of centers of cells
    x_left = 0.0
    x_right = 4.0
    y_bottom = 0.0
    y_top = 1.0

    dx = (x_right - x_left) / x_num
    dy = (y_top - y_bottom) / y_num
    x_interface = np.linspace(x_left, x_right, x_num + 1)
    y_interface = np.linspace(y_bottom, y_top, y_num + 1)
    [x_cell, y_cell] = get_cell_location(x_interface, y_interface)

    return [dx, dy, x_cell, y_cell]
