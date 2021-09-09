import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator
from scipy.spatial.transform import Rotation

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def scalar_projection(A, B):
    """
    The component of A that points in the direction of
    another vector, B
    """
    return np.tensordot(A, B, axes=1)/np.linalg.norm(B)

def JONSWAP(omega):
    Y = np.exp(-((0.191*omega-1)/np.sqrt(2)/0.07)**2)
    return 3.3**Y*155*omega**(-5)*np.exp(-944*omega**(-4))

def calc_aj(wj, dw):
    return np.sqrt(2*JONSWAP(wj)*dw)

t = 0
g = 1
domega = 0.1
omega = np.linspace(2, 12, int((12-2)/domega), dtype=float)
k = omega**2/g
eps = 2*np.pi*np.random.random(len(omega))
a = calc_aj(omega, domega)
x = np.linspace(0, np.pi, 10000)

N = 1000
X = np.linspace(-1, 1, N)
Y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(X, Y)
Z = np.ones(X.shape)
free_surface = np.array([X, Y, Z])
unit_X = np.array([1,0,0])
rotation = Rotation.from_euler("Z", np.random.normal(0, 10, len(omega)), degrees=True)

distance = [scalar_projection(free_surface.T, rotation[j].apply(unit_X)) for j in range(len(omega))]
eta = [a[j] * np.cos(k[j]*distance[j] - omega[j]*t +eps[j]) for j in range(len(omega))]
eta = np.sum(eta, axis=0)


# distance = scalar_projection(free_surface.T, unit_X)
# Z = np.cos(distance)
fig, ax = plt.subplots(subplot_kw={"projection":"3d"})

# Plot the surface
surf = ax.plot_surface(X, Y, eta, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)

# ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter('{x:0.2f}')
set_axes_equal(ax)
plt.show()