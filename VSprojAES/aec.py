import numpy as np
import math

def differ(A, B, nmode, mu, pi, lbd, u_m, Rx):

    Ap = B
    Bp = np.zeros(nmode+1)

    for n in range(1, nmode + 1):

        v1 = 0.0
        v2 = 0.0
        v3 = 0.0
        v4 = 0.0
        v5 = 0.0

        v1 = A[n] * (n*pi) ** 4 / 2

        for r in range(1, nmode + 1):
            v2 += A[r] ** 2 * (r*pi)**2 / 2
        v2 = 6 * (1-mu**2) * v2 * A[n] * (n*pi)**2 / 2

        v3 = Rx * A[n] * (n*pi)**2 / 2

        for m in range(1, nmode + 1):
            if m != n:
                v4 += (float(n * m) / (n**2 - m**2)) * (1 - (-1)**(m+n)) * A[m]
        v4 = lbd * v4

        v5 = lbd * math.sqrt(u_m / lbd) * B[n] / 2

        Bp[n] = -2.0 * (v1 + v2 + v3 + v4 + v5)

    return (Ap, Bp)

def DowellModel(f, prmt):
    # f: opened file
    lbd = prmt.lbd
    Rx = prmt.Rx
    mu = prmt.mu
    u_m = prmt.mu/prmt.Mach
    endtime = prmt.endTime * math.sqrt(prmt.D / prmt.m / prmt.a**4)
    dt = 0.001 # delta time step
    max_step = int(endtime/dt)
    w = np.zeros(max_step)
    wp = np.zeros(max_step)
    
    nmode = 6

    Anew = Bnew = A = B = np.zeros(nmode+1)
    B[1] = 0.1
    Ak1 = Ak2 = Ak3 = Ak4 = np.zeros(nmode+1)
    Bk1 = Bk2 = Bk3 = Bk4 = np.zeros(nmode+1)
    A_all = np.zeros([nmode+1, max_step])
    B_all = np.zeros([nmode+1, max_step])

    
    pi = math.pi
    time = 0

    for step in range(0, max_step):
        print time
        time = time + dt

        for n in range(1, nmode + 1):
            w[step] += A[n] * math.sin(n * 0.75 * pi)
            wp[step] += B[n] * math.sin(n * 0.75 * pi)

        #if step >= 0.1 * max_step:
        f.write('%f %f \n' % (w[step], wp[step]))

        A_all[:, step] = A
        B_all[:, step] = B

        # Runge-Kutta
        (Ak1, Bk1) = differ(A, B, nmode, mu, pi, lbd, u_m, Rx)
        (Ak2, Bk2) = differ(A + Ak1*dt/2, B + Bk1*dt/2, nmode, mu, pi, lbd, u_m, Rx)
        (Ak3, Bk3) = differ(A + Ak2*dt/2, B + Bk2*dt/2, nmode, mu, pi, lbd, u_m, Rx)
        (Ak4, Bk4) = differ(A + Ak3*dt, B + Bk3*dt, nmode, mu, pi, lbd, u_m, Rx)
        Anew = A + (Ak1 + 2*Ak2 + 2*Ak3 + Ak4) * dt / 6
        Bnew = B + (Bk1 + 2*Bk2 + 2*Bk3 + Bk4) * dt / 6

        ## forward Euler
        #(Ak1, Bk1) = differ(A, B, nmode)
        #Anew = A + Ak1 * dt
        #Bnew = B + Bk1 * dt

        # step
        A = Anew
        B = Bnew
    return w, wp


