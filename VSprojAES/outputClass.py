import numpy as np
from Euler import get_Mach, get_Ptotal, subSonicIn, subSonicOut, bumpWall, slipWall

class outputClass():

    counterFF = 1 #FF: counter for flow field writer
    counterS = 1 # counter for structure writer
    counterPP = 1 # couter for phase plane plot
    counterPrs = 1 # pressure
    counterPiston = 1 # piston theory result
    counterMach = 1 # Mach number on plate
    counterEvery = 1

    def writeFlowField(self, f, mesh, p_cell, u_cell, p_total, Mach):
        # write header
        if self.counterFF == 1:
            f.write('title = "contour"\n')
            f.write('variables = "x", "y", "pressure", "rho", "rhou", "rhov", "rhoE", "total pressure", "Mach number" \n')
        # write data zone
        f.write('\n')
        f.write('zone i = {:d}   j = {:d}   f = point\n'.format(mesh.y_num+2, mesh.x_num+2))
        for i in range(0, mesh.x_num + 2):
            for j in range(0, mesh.y_num + 2):
                f. write('{:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f}\n'.format(mesh.x_cell[i,j], mesh.y_cell[i,j], p_cell[i,j], u_cell[0,i,j], u_cell[1,i,j], u_cell[2,i,j], u_cell[3,i,j], p_total[i,j], Mach[i,j]))
        self.counterFF += 1

    def writeStructure(self, f, mesh):
        # write header
        if self.counterS == 1:
            f.write('title = "contour"\n')
            f.write('variables = "x", "w"')
        # write data zone
        n = mesh.x_num + 2
        f.write('\n')
        f.write('zone i = {:d}   f = point\n'.format(n))
        for i in range(0, n):
            f. write('{:f} {:f} \n'.format(mesh.x_cell[i,0], mesh.w_bot[i]))
        self.counterS += 1

    def writePhasePlane(self, f, w, wp):
        # write header
        if self.counterPP == 1:
            f.write('title = "contour"\n')
            f.write('variables = "w", "wp" \n')
        f.write('{:f} {:f} \n'.format(w, wp))
        self.counterPP += 1

    def writePressure(self, f, mesh, p):
        # write header
        if self.counterPrs == 1:
            f.write('title = "pressure"\n')
            f.write('variables = "x", "pressure" \n')
        # write data zone
        n = mesh.x_num + 2
        f.write('\n')
        f.write('zone i = {:d}   f = point\n'.format(n))
        for i in range(0, n):
            f. write('{:f} {:f} \n'.format(mesh.x_cell[i,0], p[i]))
        self.counterPrs += 1

    def writeMach(self, f, mesh, Machlow, Machup):
        # write header
        if self.counterMach == 1:
            f.write('title = "contour"\n')
            f.write('variables = "x", "Machlow", "MachUp" \n')
        # write data zone
        n = mesh.x_num + 2
        f.write('\n')
        f.write('zone i = {:d}   f = point\n'.format(n))
        for i in range(0, n):
            f. write('{:f} {:f} {:f} \n'.format(mesh.x_cell[i,0], Machlow[i], Machup[i]))
        self.counterMach += 1

    def writePiston(self, f, prmt, mesh):
        # write header
        if self.counterPiston == 1:
            f.write('title = "contour"\n')
            f.write('variables = "x", "pressure" \n')
        # write data zone
        n = mesh.x_num + 2
        f.write('\n')
        f.write('zone i = {:d}   f = point\n'.format(n))
        p = prmt.rho * prmt.u ** 2 / prmt.Mach * (mesh.wx_bot)
        for i in range(0, n):
            f. write('{:f} {:f} \n'.format(mesh.x_cell[i,0], p[i]))
        self.counterPiston += 1

    def writeEvery(self, fluid, flag, nRx, lbd, u_cell, p_cell, mesh, prmt, A, B, nmode, w, wp):
        if self.counterEvery == 1:

            #self.fileFF = open(''.join(['lbd', str(int(lbd)), '-Rx', str(nRx), '-flowfield.dat']), 'w')
            #self.fileMach = open(''.join(['lbd', str(int(lbd)), '-Rx', str(nRx), '-MachOnPlate.dat']), 'w')
            #self.filePrs = open(''.join(['lbd', str(int(lbd)), '-Rx', str(nRx), '-pressureOnPlate.dat']), 'w')
            self.filePP = open(''.join(['lbd', str(int(lbd)), '-Rx', str(nRx), '-phasePlane.dat']), 'w')
            #self.fileS = open(''.join(['lbd', str(int(lbd)), '-Rx', str(nRx), '-structure.dat']), 'w')

        if flag == 1:
            if fluid == 'Euler':
                # reconstruction
                u_cell[:, 0, :] = subSonicIn(u_cell[:, 1, :], prmt)# left boundary, supersonic outlet
                u_cell[:, -1, :] = subSonicOut(u_cell[:, -2, :], prmt)# right boundary, sub-sonic outlet
                u_cell[:, :, 0] = bumpWall(u_cell[:, :, 1], mesh, prmt)# lower boundary, bump + slip wall
                u_cell[:, :, -1] = slipWall(u_cell[:, :, -2]) # upper boundary, slip wall
                Mach = get_Mach(u_cell, p_cell)
                ptotal = get_Ptotal(p_cell, Mach, prmt.gamma)
                self.writeFlowField(self.fileFF, mesh, p_cell-prmt.p, u_cell, ptotal-prmt.ptotal, Mach)
                self.writeMach(self.fileMach, mesh, Mach[:,1], Mach[:, -2])
            self.writePressure(self.filePrs, mesh, p_cell[:,1]-prmt.p)
            self.writePhasePlane(self.filePP, w, wp)
            self.writeStructure(self.fileS, mesh)
        if flag == 2:
            self.writePhasePlane(self.filePP, w, wp)
        if flag == 3:
            Mach = get_Mach(u_cell, p_cell)
            ptotal = get_Ptotal(p_cell, Mach, prmt.gamma)
            self.writeMach(self.fileMach, mesh, Mach[:,1], Mach[:, -2])
            self.writePressure(self.filePrs, mesh, p_cell[:,1]-prmt.p)
            self.writeStructure(self.fileS, mesh)

        self.counterEvery += 1