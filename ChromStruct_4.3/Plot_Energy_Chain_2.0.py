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
tm=[]
cm=[]
##filen=raw_input('Energy: file name = ? ')      # Read data file name
##
##try:
##    hm=np.loadtxt(filen) # Read data
##    pylab.plot(hm[:,0],label='Data penalty')
##    pylab.plot(hm[:,1],label='Constraint penalty')
##    pylab.plot(hm[:,2],label='Total penalty')
##    pylab.legend()
##    pylab.show()
##
##except:
##    print('No such file: no data read\n')
##    sys.exit(0)


filen=input('Chain: file name = ? ')      # Read data file name


try: hm3=np.loadtxt(filen)                     # Read data
except:
    print('No such file: no data read\n')
    sys.exit(0)

filen1=input('Chain: centromeres file name = ? ')      # Read data file name
try: cm=np.loadtxt(filen1)                     # Read data
except:
    print('No such file: no data read\n')
#    sys.exit(0)

centrm=np.matrix(cm)

filen2=input('Chain: telomeres file name = ? ')      # Read data file name
try: tm=np.loadtxt(filen2)                     # Read data
except:
    print('No such file: no data read\n')
#    sys.exit(0)

telom=np.matrix(tm)

#ball=raw_input('Balls? (y/n) ')

ball='n'

if len(hm3)%3:
    print("no regular subchain data")
    sys.exit(0)

hm=np.zeros([int(len(hm3)/3),4])
for i in range(int(len(hm3)/3)):
    hm[i,:]=hm3[3*i+1,:]

fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.set_aspect('equal')
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
tck,u=interpolate.splprep(data,s=0)

#here we generate the new interpolated dataset, 
#increase the resolution by increasing the spacing, 1000 in this example
new=interpolate.splev(np.linspace(0,1,1000),tck)

ax.plot(new[0],new[1],new[2],'r',linewidth=2)
scala=1000/len(hm)
ax.plot(new[0][0:2],new[1][0:2],new[2][0:2],'k',linewidth=5)
ax.plot(new[0][998:1000],new[1][998:1000],new[2][998:1000],'purple',linewidth=5)
sL=0
for i in range(len(hm)):
    sL=sL+hm[i,3]
if np.size(centrm)>0:
    for i in range(np.size(centrm)/2):
        sC1=0
        sC2=0
        for j in range(int(centrm[i,0])):
            sC1=sC1+hm[j,3]
        C1=int(sC1*1000/sL)
        for j in range(int(centrm[i,1])):
            sC2=sC2+hm[j,3]
        C2=int(sC2*1000/sL)
        ax.plot(new[0][C1-2:C2+2],new[1][C1-2:C2+2],new[2][C1-2:C2+2],'green',linewidth=15)

if np.size(telom)>0:
    for i in range(np.size(telom)/2):
        sT1=0
        sT2=0
        for j in range(int(telom[i,0])):
            sT1=sT1+hm[j,3]
        T1=int(sT1*1000/sL)
        for j in range(int(telom[i,1])):
            sT2=sT2+hm[j,3]
        T2=int(sT2*1000/sL)
        ax.plot(new[0][T1:T2],new[1][T1:T2],new[2][T1:T2],'blue',linewidth=15)

#ax.plot(data[0],data[1],data[2],'b',linewidth=2) # (Display original points)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
#set_axes_equal(ax)
plt.show()
