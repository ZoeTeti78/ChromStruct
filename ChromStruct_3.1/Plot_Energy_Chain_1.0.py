#
##    READS A CONFIGURATION AS AN (nBeads X 4) ARRAY AND PLOTS IT
##    AS A 3D GRAPH OF nBeads CONNECTED SPHERES WITH RADIUSES hm[iBead][4] 
#


import numpy as np
from io import StringIO
from mpl_toolkits.mplot3d import Axes3D
import pylab
import matplotlib.pyplot as plt
import sys
from itertools import cycle
import copy
from scipy import interpolate

# FUNCTION drawSphere
#
# Plots each bead as a sphere 
def drawSphere(xCenter, yCenter, zCenter, r):
    #draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x=np.cos(u)*np.sin(v)
    y=np.sin(u)*np.sin(v)
    z=np.cos(v)
    # shift and scale sphere
    x = r*x + xCenter
    y = r*y + yCenter
    z = r*z + zCenter
    return (x,y,z)

# FUNCTION set_axes_equal
#
# Equalizes the cartesian axes in display
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

    x_range = x_limits[1] - x_limits[0]; x_mean = np.mean(x_limits)
    y_range = y_limits[1] - y_limits[0]; y_mean = np.mean(y_limits)
    z_range = z_limits[1] - z_limits[0]; z_mean = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_mean - plot_radius, x_mean + plot_radius])
    ax.set_ylim3d([y_mean - plot_radius, y_mean + plot_radius])
    ax.set_zlim3d([z_mean - plot_radius, z_mean + plot_radius])


## MAIN

hm=[]
hm3=[]
filen=raw_input('Energy: file name = ? ')      # Read data file name

try:
    hm=np.loadtxt(filen) # Read data
    pylab.plot(hm[:,0],label='Data penalty')
    pylab.plot(hm[:,1],label='Constraint penalty')
    pylab.plot(hm[:,2],label='Total penalty')
    pylab.legend()
    pylab.show()

except:
    print 'No such file: no data read\n'
#    sys.exit(0)


filen=raw_input('Chain: file name = ? ')      # Read data file name

try: hm3=np.loadtxt(filen)                     # Read data
except:
    print 'No such file: no data read\n'
    sys.exit(0)

#ball=raw_input('Balls? (y/n) ')

ball='n'

if len(hm3)%3:
    print "no regular subchain data"
    sys.exit(0)

hm=np.zeros([len(hm3)/3,4])
for i in range(len(hm3)/3):
    hm[i,:]=hm3[3*i+1,:]

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect('equal')
col_gen=cycle('bgrcmyk')
for i in range(len(hm)):
    x=hm[i][0]
    y=hm[i][1]
    z=hm[i][2]
    r=hm[i][3]
    if ball[0]=='y':
        (xs,ys,zs)=drawSphere(x,y,z,r)
        ax.plot_wireframe(xs,ys,zs,color=col_gen.next())
#
#  Display smooth curve
#

data=hm[:,0:3]
data=data.transpose() # we need to transpose to get the data in the right format

#now we get all the knots and info about the interpolated spline
tck,u=interpolate.splprep(data)

#here we generate the new interpolated dataset, 
#increase the resolution by increasing the spacing, 1000 in this example
new=interpolate.splev(np.linspace(0,1,1000),tck)

ax.plot(new[0],new[1],new[2],'r',linewidth=2)


#ax.plot(data[0],data[1],data[2],'b',linewidth=2) # (Display original points)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
set_axes_equal(ax)
plt.show()
