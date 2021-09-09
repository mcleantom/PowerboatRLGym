import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

t = 3
dt = 0.1
nt = 100
t_steps = np.arange(0, dt*nt, dt)
g = 1
a = 1
w = 1
kh = 3

def water(t):
    # deep water velocity potential
    u_x = (g*a/w)*np.cosh(kh+kz)/np.cosh(kh)*np.cos(kx-w*t)
    u_z = (g*a/w)*np.sinh(kh+kz)/np.cosh(kh)*np.sin(kx-w*t)
    
    # deep water wave elevation
    z = 0.3*a*np.cos(kx[0,:] - w*t)
    return u_x, u_z, z


kx, kz = np.meshgrid(np.arange(0, 4*np.pi, .25),
                     np.arange(-kh, 0, .25))

fig, ax = plt.subplots(1, 1)
u_x, u_z, z = water(t)
Q = ax.quiver(kx, kz, u_x, u_z)
# FS_x = np.arange(0, 4*np.pi, .25)
Z, = ax.plot(kx[0,:], z)
fig.show()

def animate(i, Q, Z, kx, kz):
    u_x, u_z, z = water(i*dt)
    
    Q.set_UVC(u_x, u_z)
    Z.set_ydata(z)

anim = animation.FuncAnimation(fig, animate, fargs=(Q, Z, kx, kz),
                                interval=dt, blit=False)
