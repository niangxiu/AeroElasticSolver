from math import pi, sqrt
from Euler import get_Ptotal

class parameter_class:
    # used to generate the boundary condition for double Mach problem
    # data:
    def __init__(self, name, endTime, Rx, lbd):
    
        self.endTime = endTime
        self.Rx = Rx
        self.lbd = lbd
        self.name = name
        self.gamma = 1.4
        gamma = 1.4
        R = 8.314 # gas constant, unit: J?K?1?mol?1
        molemass = 28.97e-3 # mass per mole (Molecular Mass) of air, unit: kg/mol
    
        if name =='bp':
            self.rhol = 8.0
            self.ul = 7.1447
            self.vl = 0
            self.pl = 116.5
            
            self.rhor = 8.0
            self.ur = 7.1447
            self.vr = 0
            self.pr = 116.5
    
            self.rhoEl = self.pl / (gamma-1) + 0.5 * self.rhol * (self.ul**2 + self.vl**2)
            self.rhoEr = self.pr / (gamma-1) + 0.5 * self.rhor * (self.ur**2 + self.vr**2)
    
        elif name == 'dm':
            self.rhol = 8.0
            self.ul = 7.1447
            self.vl = -4.125
            self.pl = 116.5
    
            self.rhor = 1.4
            self.ur = 0.0
            self.vr = 0.0
            self.pr = 1.0

            self.rhoEl = self.pl / (gamma-1) + 0.5 * self.rhol * (self.ul**2 + self.vl**2)
            self.rhoEr = self.pr / (gamma-1) + 0.5 * self.rhor * (self.ur**2 + self.vr**2)

        elif name == 'AE_dowell':
            # three arbitrary parameter
            self.rho = 1#.4
            self.a = 2.0
            self.p = 1.0/self.gamma # pressure
            # dimensionless groups (\pi)

            self.Mach = 1.2
            self.mu = 0.01 * self.Mach
            self.X = 0.75
            self.nu = 0.3
            self.P = 0.0 # dimensionless pressure
            self.eta = 0.001 # h/a
            # other parameters
            self.sound = sqrt(gamma * self.p / self.rho)
            self.u = self.Mach * self.sound
            self.v = 0.0
            self.rhoE = self.p / (gamma-1) + 0.5 * self.rho * (self.u**2 + self.v**2)          
            self.h = self.a * self.eta
            self.D = self.rho * self.u **2 * self.a**3 / (self.Mach * self.lbd)
            self.ptotal = self.p + 0.5 * self.rho * (self.u**2 + self.v**2)
            self.m = self.rho * self.a / self.mu
            self.NxE = self.Rx * self.D / self.a ** 2
            self.E = self.D * 12 * (1-self.nu**2) / self.h**3
            # for compliance with other parts of the code
            self.rhol = self.rho
            self.ul = self.u
            self.vl = self.v
            self.pl = self.p
            self.rhoEl = self.rhoE
            self.rhor = self.rho
            self.ur = self.u
            self.vr = self.v
            self.pr = self.p
            self.rhoEr = self.rhoE
            # other parameters should be unecessary.

        elif name == 'Nibump':
            # three arbitrary parameter
            self.rho = 1.4
            self.a = 2.0
            self.p = 1.0 # pressure
            # dimensionless groups (\pi)
            self.Mach = 0.675
            self.mu = 0.01 * self.Mach
            self.X = 0.75
            self.nu = 0.3
            self.P = 0.0 # dimensionless pressure in 
            self.eta = 0.001 # h/a
            # other parameters
            self.sound = sqrt(gamma * self.p / self.rho)
            self.u = self.Mach * self.sound
            self.v = 0.0
            self.rhoE = self.p / (gamma-1) + 0.5 * self.rho * (self.u**2 + self.v**2)          
            self.h = self.a * self.eta
            self.D = self.rho * self.u **3 * self.a**3 / (self.Mach * self.lbd)
            self.m = self.rho * self.a / self.mu
            self.ptotal = get_Ptotal(self.p, self.Mach, self.gamma)
            # for compliance with other parts of the code
            self.rhol = self.rho
            self.ul = self.u
            self.vl = self.v
            self.pl = self.p
            self.rhoEl = self.rhoE
            self.rhor = self.rho
            self.ur = self.u
            self.vr = self.v
            self.pr = self.p
            self.rhoEr = self.rhoE
            # other parameters should be unecessary.



        

