class boundary_class:
    # used to generate the boundary condition for double Mach problem
    # data:
    def __init__(self, flag):
    
        gamma = 1.4
    
        if flag =='bp':
            self.rhol = 8.0
            self.ul = 7.1447
            self.vl = 0
            self.pl = 116.5
            
            self.rhor = 8.0
            self.ur = 7.1447
            self.vr = 0
            self.pr = 116.5
    
    
        elif flag == 'dm':
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

