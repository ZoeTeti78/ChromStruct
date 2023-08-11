# ChromStruct v 4.3

# Reconstruction of the 3D structure of the chromatin fiber from Hi-C
# contact frequency data.

# Copyright (C) 2020 Emanuele Salerno, Claudia Caudai (ISTI-CNR Pisa)
# {emanuele.salerno,claudia.caudai}@isti.cnr.it

##This program is free software: you can redistribute it and/or modify
##it under the terms of the GNU General Public License as published by
##the Free Software Foundation, version 3 of the License.

##This program is distributed in the hope that it will be useful,
##but WITHOUT ANY WARRANTY; without even the implied warranty of
##MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##GNU General Public License for more details.

warranty="This program is distributed in the hope that it will be useful,\n"\
          +"but WITHOUT ANY WARRANTY; without even the implied warranty of\n"\
          +"MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n"\
          +"GNU General Public ffitrnaLicense version 3 for more details"\
          +"<http://www.gnu.org/licenses/>.\n\n"

conditions="This program is free software: you can redistribute it and/or modify\n"\
            +"it under the terms of the GNU General Public License as published by\n"\
            +"the Free Software Foundation, version 3 of the License"\
            +"<http://www.gnu.org/licenses/>.\n\n"



##You should have received a copy of the GNU General Public License
##along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Date 10/11/2020

# References:
# Caudai C., Salerno E., Zoppe' M., Tonazzini A.
# Inferring 3D chromatin structure using a multiscale
# approach based on quaternions.
# In: BMC Bioinformatics, vol. 16, article n. 234. BioMed Central, 2015.
# DOI: 10.1186/s12859-015-0667-0
#
# Caudai C., Salerno E., Zoppe' M., Tonazzini, A.
# Estimation of the Spatial Chromatin Structure Based on a Multiresolution Bead-Chain Model
# IEEE/ACM TCBB, 2018.
# DOI: 10.1109/TCBB.2018.2791439
#
# Caudai C., Salerno E., Zoppe' M., Merelli I., Tonazzini, A.
# CHROMSTRUCT 4: A Python Code to Estimate the Chromatin Structure from Hi-C Data.
# IEEE/ACM TCBB, 2019.
# DOI: 10.1109/TCBB.2018.2838669


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


import copy
from copy import deepcopy
import numpy as np
import math
import random
from io import StringIO
from scipy import linalg as LA
from mpl_toolkits.mplot3d import Axes3D
import pylab
import matplotlib.pyplot as plt
import datetime
import time
import sys
from itertools import cycle
from pyquaternion import Quaternion

# PARAMETERS
#==================================================================================================
#

#
# CODE VERSION
version="4.3 (Nov 10, 2020)"

#
# PENALTY FUNCTION (ENERGY) TUNING
#
# Fitness part
diagneg=2             # # neglected diagonals in contact matrix (in populate, to select the relevant pairs for data fit)
datafact=0.3           # Fraction of the pairs per block used to build the penalty function (real, in populate)

# Constraint part
energy_scale=4.    # Tunes the extent of the moderate penalty range around the threshold distance 
                           # (floating; physical dimensions 1/length; measurement unit: reciprocal of the units used for DIA)
energy_exp=5               # Tunes the slopes of the penalty in the transitions between the moderate range
                           # and the external intervals (odd integer, dimensionless)

#
# TAD EXTRACTION (FUNCTION block)
span=6                # SIZE OF THE MOVING TRIANGLE
window=3              # MOVING AVERAGE WINDOW APERTURE
minsize=6             # MINIMUM REQUIRED DIAGONAL BLOCK SIZE


#
# ANNEALING: DETERMINATION OF THE REGULARIZATION PARAMETER lamb
regulenergy=0.005       # TARGET RELATIVE WEIGHT CONSTRAINT/FITNESS
avgenergy=500        # FIXED NUMBER OF AVERAGING CYCLES TO EVALUATE THE CONSTRAINT/FITNESS RATIO
percenergy=20         # MAXIMUM ACCEPTED PERCENTILE IN THE ENERGY SAMPLE SETS (integer <= 100)

#
# ANNEALING: WARM-UP PHASE
Tmax=4000.            # Start temperature, warm-up phase
itwarm=50000          # Max # warm-up cycles
incrtemp=1.2          # Fixed temperature increase coefficient at warm-up
checkwarm=500         # Fixed milestone on warm-up cycles to check the acceptance rate
muwarm=0.9            # Minimum acceptance rate to end warm-up

#
# ANNEALING: SAMPLING PHASE
itmax=50000           # Max # annealing cycles (positive integer)
itstop=100            # # consecutive cycles with energy variations within StopTolerance to stop annealing
StopTolerance=1.e-2   # Stop tolerance
RANDPLA=2*0.05        # Max planar angle perturbation at each update (floating, radians, doubled for convenience)
RANDDIE=2*0.05        # Max dihedral angle perturbation at each update (floating, radians, doubled for convenience)
#RANDPLA=2*0.5        # Max planar angle perturbation at each update (floating, radians, doubled for convenience)
#RANDDIE=2*0.5        # Max dihedral angle perturbation at each update (floating, radians, doubled for convenience)
decrtemp=0.998        # Fixed cooling coefficient at each annealing cicle


#
# GEOMETRIC PARAMETERS
DIA=30.               # Chromatin fiber diameter (nm)
RIS=5.              # Contact matrix (full) genomic resolution (kbp)
NB=3.                 # kilobase-pairs in a chromatin fragment with length DIA #if DIA=10 NB=1
crate=0.002            # compactness rate for telomeres and centromeres  

    
NC=DIA/NB             # # length of the genomic resolution unit
LMAX=RIS**(1./3)*NC*np.sqrt(RIS)/np.pi    # hypothesized maximum size of a bead in the full-resolution model (nm)
LMIN=RIS**(1./3)*NC*np.sqrt(RIS)/(np.pi*2)     # hypothesized maximum size of a bead in the full-resolution model (nm)# hypothesized minimum bead size in the full-resolution model (nm)
LMAX=RIS+10
LMIN=RIS/2+10
NMAX=0.               # Maximum contact frequency: fixed or computed in the current data matrix (see function inizchain)
extrate=0.3           # Size tuning for beads at levels > 0

#==================================================================================================
#


#==================================================================================================
#
#===================================================================================================

#
# CLASS DEFINITIONS (SPECIFIC)
#



# CLASS bead
#
# 3 attributes class: endpoint 1 (endp1); centroid (centrd); endpoint 2 (endp2)
# 1 attribute type floating point: the bead size (ext)
#
class bead:
    def __init__(self,endp1=np.array([0.,0.,0.]),centrd=np.array([0.,0.,0.]),endp2=np.array([0.,0.,0.]),ext=DIA/2):
        self.centrd=centrd
        self.endp1=endp1
        self.endp2=endp2
        self.ext=ext
    
# Alternative string definitions
#
#  Descriptive
##    def __str__(self):
##        return          'endpoint 1: '+str(self.endp1)+'\n'+ \
##                        'centroid  : '+str(self.centrd)+ '\n'+ \
##                        'endpoint 2: '+str(self.endp2)+ '\n'+\
##                        'extent    : '+str(self.ext)+' (nm)'
##
#
#  Normal
    def __str__(self):
        return          str(self.endp1[0])+' '+ str(self.endp1[1])+' '+ str(self.endp1[2])+' '+str(self.ext)+'\n'+ \
                        str(self.centrd[0])+' '+ str(self.centrd[1])+' '+ str(self.centrd[2])+' '+str(self.ext)+'\n'+ \
                        str(self.endp2[0])+' '+ str(self.endp2[1])+' '+ str(self.endp2[2])+' '+str(self.ext)
#
#  Final (prints the centroid coordinates and the size)
    def finalstr(self):
        return          str(self.centrd[0])+' '+ str(self.centrd[1])+' '+ str(self.centrd[2])+' '+str(self.ext)
#
# Inner angle
    def innerangle(self):
        v1=self.endp1-self.centrd
        v2=self.endp2-self.centrd
        return math.acos(np.dot(v1,v2)/(LA.norm(np.float128(v1))*LA.norm(np.float128(v2))))
#        return math.acos(np.dot(v1,v2)/(LA.norm(v1)*LA.norm(v2)))
#
# Distances between the centroid and the endpoints
    def d1d2(self):
        d1=np.linalg.norm(self.endp1-self.centrd)
        d2=np.linalg.norm(self.centrd-self.endp2)
        return d1,d2
        
#
# Shifts rigidly the entire bead
    def shiftbd(self,displacement):
        self.centrd=self.centrd+displacement
        self.endp1=self.endp1+displacement
        self.endp2=self.endp2+displacement
#
# Shifts rigidly the entire bead so as to put endpoint 1 in (0.,0.,0.)
    def zerobead(self):
        self.centrd=self.centrd-self.endp1
        self.endp2=self.endp2-self.endp1
        self.endp1=np.array([0.,0.,0.])





#
# CLASS chain
#
# 1 attribute: a list of bead class objects
#

class chain:
    def __init__(self,bead=[]):
        self.bead=bead

#
# Chain length
    def __len__(self):
            return len(self.bead)

# Alternative string definitions
#
#  Descriptive (introspective)
##    def __str__(self):
##        try: string='chain: '+namefind(self)+' \n'
##        except: string='chain:\n'
##        for i in range(len(self.bead)):
##            string=string+'bead '+str(i)+'\n'+str(self.bead[i])+'\n'
##        return string
#
#  Normal
    def __str__(self):
        stringa= ''
        for i in range(len(self)):
            stringa=stringa+str(self.bead[i])+'\n'
        return stringa
#
#  Final (prints the bead list through bead.finalstr
    def finalstr(self):
        stringa= ''
        for i in range(len(self)):
            stringa=stringa+self.bead[i].finalstr()+'\n'
        return stringa

#
# Appends a bead to the chain
    def chainadd(self,newbead): # aggiunta di una singola palla alla catena
        self.bead.append(newbead)

#
# Appends an entire chain
    def chainmerge(self,appchain): # accoda una nuova catena alla catena corrente
        for i in range(len(appchain)):
            self.chainadd(appchain.bead[i])
#
# Arranges the chain beads rigidly so as to put endp1 of each bead on endp2 of the immediate predecessor
    def formchain(self):
        for i in range(1,len(self)):
            self.bead[i].shiftbd(self.bead[i-1].endp2-self.bead[i].endp1)

            
#
# Shifts the entire chain rigidly so as to put endp1 of the first bead in (0.,0.,0.)
    def zerochain(self):
        displacement=-1*self.bead[0].endp1
        for i in range(len(self)):
            self.bead[i].shiftbd(displacement)
#
# Shifts the entire chain rigidly so as to put endp1 of the first chain in a specified location
    def shiftchain(self,startpoint):
        for i in range(len(self.bead)):
            self.bead[i].shiftbd(startpoint)

#
# Using quaternions, rotates bead(ibead) of an angle in the plane generated by (endp2-endp1)(ibead-1) and (endp2-endp1)(ibead)
# (if these two vectors are colinear, the bead is rotated in the plane z=0.)
    def quaplanar(self,ibead,anglequa):
        axisqua=np.cross(self.bead[ibead].endp2-self.bead[ibead].endp1,self.bead[ibead-1].endp2-self.bead[ibead-1].endp1)
        if (axisqua[0]==0. and axisqua[1]==0. and axisqua[2]==0.):
            qua=Quaternion(axis=[0.,0.,1.],angle=anglequa).normalised
        else:
            qua=Quaternion(axis=[axisqua[0],axisqua[1],axisqua[2]],angle=anglequa).normalised
        return qua, axisqua
            
#
# Using quaternions, modifies the dihedral angle between the ibead-th and the (ibead-1)-th beads
    def quadihedral(self,ibead,anglequa):
        axisqua=self.bead[ibead-1].endp2-self.bead[ibead-1].endp1
        qua=Quaternion(axis=[axisqua[0],axisqua[1],axisqua[2]],angle=anglequa).normalised
        return qua
            

#
# Random chain perturbation through a single bead (in this version, d1 and d2 are kept fixed)
    def perturb(self,ibead):
        ch=deepcopy(self)
        deltap=(random.random()-0.5)*RANDPLA
        deltad=(random.random()-0.5)*RANDDIE
##        deltap=(-0.5)*RANDPLA
##        deltad=RANDDIE
        [quap, axis]=ch.quaplanar(ibead,deltap)
        quad=ch.quadihedral(ibead,deltad)
        if np.linalg.norm(axis):
            for i in range(ibead,len(self)):
                a=ch.bead[i]
##                print(LA.norm(a.centrd))
##                print(LA.norm(quap.rotate(a.centrd)))
                a.zerobead()
                ch.bead[i].endp1=ch.bead[i-1].endp2
                ch.bead[i].centrd=quap.rotate(a.centrd)+ch.bead[i-1].endp2
                ch.bead[i].endp2=quap.rotate(a.endp2)+ch.bead[i-1].endp2
        else:
            for i in range(ibead,len(self)):
                a=ch.bead[i]
                a.zerobead()
                a.centrd=np.array([a.centrd[0]*math.cos(deltap)-a.centrd[1]*math.sin(deltap),\
                                 a.centrd[0]*math.sin(deltap)+a.centrd[1]*math.cos(deltap),a.centrd[2]])
                a.endp2=np.array([a.endp2[0]*math.cos(deltap)-a.endp2[1]*math.sin(deltap),\
                                 a.endp2[0]*math.sin(deltap)+a.endp2[1]*math.cos(deltap),a.endp2[2]])
                ch.bead[i].endp1=ch.bead[i-1].endp2
                ch.bead[i].centrd=a.centrd+ch.bead[i-1].endp2
                ch.bead[i].endp2=a.endp2+ch.bead[i-1].endp2
                
        for i in range(ibead,len(self)):
            b=ch.bead[i]
##            print(LA.norm(b.centrd))
##            print(LA.norm(quad.rotate(b.centrd)))
            b.zerobead()
            ch.bead[i].endp1=ch.bead[i-1].endp2
            ch.bead[i].centrd=quad.rotate(b.centrd)+ch.bead[i-1].endp2
            ch.bead[i].endp2=quad.rotate(b.endp2)+ch.bead[i-1].endp2

        return ch


#===================================================================================================

#
# FUNCTION DEFINITIONS:
#  - centr (finds centromeres and telomers in the contact matrix)
#  - distmatrix (computes the distance matrix given a chain)
#  - block (extract approximate diagonal blocks given a matrix)
#  - populate (populates the list of relevant pairs given a matrix)
#  - namefind (finds an object name in the globals dictionary)
#  - checkstop (check the stop criterion for annealing)
#  - cool (lowers the temperature during annealing)
#  - warm (raises the temperature during warm-up)
#  - checktest (checks the acceptance criterion during annealing)
#  - fitcomp (computes the data fit penalty)
#  - constrcomp (computes the constraint penalty)
#  - energy (computes the complete penalty function)
#  - annealing (simulated annealing procedure)
#  - skipannealing (skip annealing in case of telomer or centromere)
#  - binning (bins a matrix given the sizes of the relevant diagonal blocks)
#  - compbead (computes an equivalent low-resolution bead given a reconstructed subchain)
#  - savepar (saves output parameters to file)
#  - saveconf (saves output configuration to file)
#  - constr (computes the initial beas sizes given the geometric parameters)
#  - centrsize (computes the initial beas sizes of centromere and telomeres)
#  - inizchain (computes an initial-guess chain)
#  - align (alings a subchain to its lower-resolution bead counterpart)
#  - compose (reconstructs the full-resolution chain from the lower-resolution subchains)
#  - drawSphere (plots each bead as a sphere)
#  - set_axes_equal (equalizes axis scales in display)
#  - output (plots and saves the final configuration to file)
#  - chromstruct (overall recursive procedure)
#



# FUNCTION centr
#
# Test to find centromeres and telomers in the contact matrix

def centr(hm):
    centromere=[]
    lc=[]
    telomeres=[]
    i=0
    while i<(len(hm)-minsize):
        k=0
        while np.max(hm[:,i])==0. and i<len(hm)-1:
            k=k+1
            i=i+1
        if k>=minsize:
            if i-k==0:
                telomeres.append([i-k,i-1])
            elif i==len(hm)-1:
                telomeres.append([i-k,i])
            else:
                centromere.append([i-k,i-1])
                lc.append(k-1)
        i=i+1
    if len(centromere)>1:
        for s in range(len(centromere)):
            mlc=max(lc)
            if (centromere[s][1]-centromere[s][0])==mlc:
                centromere=[[centromere[s][0],centromere[s][1]]]
                break   ###
    bigblocks=[]

    if len(telomeres)>0:
        if telomeres[0][0]==0:
            mat=np.zeros([telomeres[0][1]+1,telomeres[0][1]+1])
            bigblocks.append(mat)
            if len(centromere)>0:
                mat=np.zeros([centromere[0][0]-telomeres[0][1]-1,centromere[0][0]-telomeres[0][1]-1])
                for j in range(centromere[0][0]-telomeres[0][1]-1):
                    for s in range(centromere[0][0]-telomeres[0][1]-1):
                        mat[j,s]=hm[telomeres[0][1]+j+1,telomeres[0][1]+s+1]
                bigblocks.append(mat)
                mat=np.zeros([centromere[0][1]-centromere[0][0]+1,centromere[0][1]-centromere[0][0]+1])
                bigblocks.append(mat)
                if len(telomeres)<2:
                    mat=np.zeros([len(hm)-centromere[0][1]-1,len(hm)-centromere[0][1]-1])
                    for j in range(len(hm)-centromere[0][1]-1):
                        for s in range(len(hm)-centromere[0][1]-1):
                            mat[j,s]=hm[centromere[0][1]+j+1,centromere[0][1]+s+1]
                    bigblocks.append(mat)
                else:
                    mat=np.zeros([telomeres[1][0]-centromere[0][1]-1,telomeres[1][0]-centromere[0][1]-1])
                    for j in range(telomeres[1][0]-centromere[0][1]-1):
                        for s in range(telomeres[1][0]-centromere[0][1]-1):
                            mat[j,s]=hm[centromere[0][1]+j+1,centromere[0][1]+s+1]
                    bigblocks.append(mat)
                    mat=np.zeros([telomeres[1][1]-telomeres[1][0]+1,telomeres[1][1]-telomeres[1][0]+1])
                    bigblocks.append(mat)
            else:
                if len(telomeres)<2:
                    mat=np.zeros([len(hm)-telomeres[0][1]-1,len(hm)-telomeres[0][1]-1])
                    for j in range(len(hm)-telomeres[0][1]-1):
                        for s in range(len(hm)-telomeres[0][1]-1):
                            mat[j,s]=hm[telomeres[0][1]+j+1,telomeres[0][1]+s+1]
                    bigblocks.append(mat)
                else:
                    mat=np.zeros([telomeres[1][0]-telomeres[0][1]-1,telomeres[1][0]-telomeres[0][1]-1])
                    for j in range(telomeres[1][0]-telomeres[0][1]-1):
                        for s in range(telomeres[1][0]-telomeres[0][1]-1):
                            mat[j,s]=hm[telomeres[0][1]+j+1,telomeres[0][1]+s+1]
                    bigblocks.append(mat)
                    mat=np.zeros([telomeres[1][1]-telomeres[1][0]+1,telomeres[1][1]-telomeres[1][0]+1])
                    bigblocks.append(mat)
        else:
            if len(centromere)>0:
                mat=np.zeros([centromere[0][0],centromere[0][0]])
                for j in range(centromere[0][0]):
                    for s in range(centromere[0][0]):
                        mat[j,s]=hm[j,s]
                bigblocks.append(mat)
                mat=np.zeros([centromere[0][1]-centromere[0][0]+1,centromere[0][1]-centromere[0][0]+1])
                bigblocks.append(mat)
                mat=np.zeros([telomeres[0][0]-centromere[0][1]-1,telomeres[0][0]-centromere[0][1]-1])
                for j in range(telomeres[0][0]-centromere[0][1]-1):
                    for s in range(telomeres[0][0]-centromere[0][1]-1):
                        mat[j,s]=hm[centromere[0][1]+j+1,centromere[0][1]+s+1]
                bigblocks.append(mat)
                mat=np.zeros([telomeres[0][1]-telomeres[0][0]+1,telomeres[0][1]-telomeres[0][0]+1])
                bigblocks.append(mat)
            else:
                mat=np.zeros([telomeres[0][0],telomeres[0][0]])
                for j in range(telomeres[0][0]):
                    for s in range(telomeres[0][0]):
                        mat[j,s]=hm[j,s]
                bigblocks.append(mat)
                mat=np.zeros([telomeres[0][1]-telomeres[0][0]+1,telomeres[0][1]-telomeres[0][0]+1])
                bigblocks.append(mat)
    else:
        if len(centromere)>0:
            mat=np.zeros([centromere[0][0],centromere[0][0]])
            for j in range(centromere[0][0]):
                for s in range(centromere[0][0]):
                    mat[j,s]=hm[j,s]
            bigblocks.append(mat)
            mat=np.zeros([centromere[0][1]-centromere[0][0]+1,centromere[0][1]-centromere[0][0]+1])
            bigblocks.append(mat)
            mat=np.zeros([len(hm)-centromere[0][1]-1,len(hm)-centromere[0][1]-1])
            for j in range(len(hm)-centromere[0][1]-1):
                for s in range(len(hm)-centromere[0][1]-1):
                    mat[j,s]=hm[centromere[0][1]+j+1,centromere[0][1]+s+1]
            bigblocks.append(mat)
        else:
            bigblocks.append(hm)
    return centromere, telomeres, bigblocks


# FUNCTION distmatrix
#
# Distance matrix given a chain configuration (distances between centroids; returns an upper-triangular matrix)
def distmatrix(conf):
    M=np.zeros([len(conf),len(conf)])
    for i in range(len(conf)):
        for j in range(i+1,len(conf)):
            M[i][j]=np.linalg.norm(conf.bead[j].centrd-conf.bead[i].centrd)
    return M
             


# FUNCTION block (New version, after Mizuguchi et al. 2014)
#
# TAD detection (approximate diagonal block extraction from contact matrix) for level
def block(level,contacts,window=3,minsize=6,span=6):
    if level==0:
        blocks=[]
        bounds=[0]
        sizes=[]
        for s in range(len(bigblocks)):
            bigblock=bigblocks[s]

    # Compute the moving cumulative sum over an off-diagonal triangle of size 'span'
            if bigblock.max()==0:
                subblocks=[bigblock-1]
                subbounds=[0,len(bigblock)]
                subsizes=[len(bigblock)]
                
            else:
                avg=[]

                for i in range(len(bigblock)-span):
                    cum=0
                    for j in range(i,i+span):
                            for k in range(j+1,i+span):
                                    cum=cum+bigblock[j,k]
                    avg.append(cum)

            # Smooth 'avg' vector through moving average of aperture 'window'

                smavg=[]

                for i in range(len(avg)-window):
                    smavg.append(np.mean(avg[i:i+window]))

            # Locate potential block bounds as local minima of 'smavg'

                subbounds=[0]                             # initialize block bounds list
                for i in range(1,len(smavg)-1):
                    if smavg[i]<=smavg[i-1] and smavg[i]<=smavg[i+1]: subbounds.append(i)

                subbounds.append(len(bigblock))

            # Merge blocks smaller than 'minsize'

                i=1
                while i in range(1,len(subbounds)) and len(subbounds)>2:
                    if subbounds[1]<minsize: del subbounds[1]
                    if subbounds[i]-subbounds[i-1]<minsize:
                        if smavg[subbounds[i]]>=smavg[subbounds[i-1]] or i==1:
                            del subbounds[i]
                        else:
                            del subbounds[i-1]
                    else:
                        i=i+1
                if subbounds[-1]-subbounds[-2]<minsize and len(subbounds)>2:
                    del subbounds[-2]


            ## Required minimum number of blocks

                if len(subbounds)<=minsize:
                    for i in range(1,len(subbounds)-1):
                        del subbounds[1]
                


            # Compute block sizes

                subsizes=[]

                for i in range(len(subbounds)-1):
                    subsizes.append(subbounds[i+1]-subbounds[i])

            # Fill the block list
                
                subblocks=[]
                for i in range(len(subbounds)-1):
                    subblocks.append(bigblock[subbounds[i]:subbounds[i+1],subbounds[i]:subbounds[i+1]])
            for h in range(len(subblocks)):
                blocks.append(subblocks[h])
            sizes=np.concatenate((sizes,subsizes))
            fb=0
            for p in range(1,s+1):
                fb=fb+len(bigblocks[p-1])
            for h in range(len(subbounds)):
                subbounds[h]=subbounds[h]+fb
            
            bounds=np.concatenate((bounds,subbounds[1:len(subbounds)]))
    else:
    # Compute the moving cumulative sum over an off-diagonal triangle of size 'span'

        avg=[]

        for i in range(len(contacts)-span):
            cum=0
            for j in range(i,i+span):
                    for k in range(j+1,i+span):
                            cum=cum+contacts[j,k]
            avg.append(cum)

    # Smooth 'avg' vector through moving average of aperture 'window'

        smavg=[]

        for i in range(len(avg)-window):
            smavg.append(np.mean(avg[i:i+window]))

    # Locate potential block bounds as local minima of 'smavg'

        bounds=[0]                             # initialize block bounds list
        for i in range(1,len(smavg)-1):
            if smavg[i]<=smavg[i-1] and smavg[i]<=smavg[i+1]: bounds.append(i)

        bounds.append(len(contacts))

    # Merge blocks smaller than 'minsize'

        i=1
        while i in range(1,len(bounds)) and len(bounds)>2:
            if bounds[1]<minsize: del bounds[1]
            if bounds[i]-bounds[i-1]<minsize:
                if smavg[bounds[i]]>=smavg[bounds[i-1]] or i==1:
                    del bounds[i]
                else:
                    del bounds[i-1]
            else:
                i=i+1
        if bounds[-1]-bounds[-2]<minsize and len(bounds)>2:
            del bounds[-2]


    ## Required minimum number of blocks

        if len(bounds)<=minsize:
            for i in range(1,len(bounds)-1):
                del bounds[1]
        


    # Compute block sizes

        sizes=[]

        for i in range(len(bounds)-1):
            sizes.append(bounds[i+1]-bounds[i])

    # Fill the block list
        
        blocks=[]
        for i in range(len(bounds)-1):
            blocks.append(contacts[bounds[i]:bounds[i+1],bounds[i]:bounds[i+1]])
        

    return blocks,sizes,bounds # three lists are returned


# FUNCTION block_def (with predefined blocks)
#
# TAD detection (approximate diagonal block extraction from contact matrix) for level
def block_def(level,contacts,window=3,minsize=6,span=6):
    if level==0:
        bounds=[]
        sizes=[]
        for i in range(len(bounds)-1):
            sizes.append(bounds[i+1]-bounds[i])
        # Fill the block list    
        blocks=[]
        for i in range(len(bounds)-1):
            blocks.append(contacts[bounds[i]:bounds[i+1],bounds[i]:bounds[i+1]])
    if level==1:
        bounds=[]
        sizes=[]
        for i in range(len(bounds)-1):
            sizes.append(bounds[i+1]-bounds[i])
        # Fill the block list    
        blocks=[]
        for i in range(len(bounds)-1):
            blocks.append(contacts[bounds[i]:bounds[i+1],bounds[i]:bounds[i+1]])
        
    return blocks,sizes,bounds # three lists are returned
    


    

# FUNCTION populate
#
# Returns a list of 2-tuples specifying the bead pairs with relevant contact numbers 
def populate(mat):
    string= "block size "+str(len(mat))+'\n'
    print(string)
    outdata.write(string)
    matrix=mat-np.diag(np.diag(mat)) # Delete the diagonals to be neglected
    for ineg in range(1,diagneg): 
        matrix=matrix-np.diag(np.diag(mat,ineg),ineg)
    L=[(0,len(matrix)-1)]
    # This strategy selects a fraction datafact of the total number of pairs in the block
    c=sorted(matrix.reshape([len(matrix)**2]),reverse=True)
#    print(c)
    for i in range(len(matrix)):
        for j in range(i+diagneg,len(matrix)):
#            print("matrix[i,j]", matrix[i,j], i, j)
#            print(c[int(0.5*len(matrix)*(len(matrix)-1)*datafact)])
            if matrix[i,j]>c[int(0.5*len(matrix)*(len(matrix)-1)*datafact)]: 
                L.append((i,j))
    string = "number of relevant pairs in this block: "+str(len(L))+str("(datafact = "+str(datafact)+")\n")
    print(string)
    outdata.write(string)
    return L


### FUNCTION namefind
###
### Introspection (used to find the chain name in Descriptive __str__ of class chain)
##def namefind(obj):
##    for i in range(len(globals())):
##        if id(globals().items()[i][1])==id(obj):
##            return globals().items()[i][0]
##


# FUNCTION checkstop
#
# Stop criterion for simulated annealing
def checkstop(series,it):
    return (len(series)>itstop and np.array(series[-itstop:-1]).max()-np.array(series[-itstop:-1]).min()\
           <=np.array(series[-itstop:-1]).max()*StopTolerance) or it>itmax


# FUNCTION cool
#
# Cooling in simulated annealing
def cool(T):
    return decrtemp*T


# FUNCTION warm
#
# Temperature increase in annealing warm-up
def warm(T,AcceptRate):
    try:
        T2=incrtemp*T/AcceptRate**1.5
    except:
        T2=incrtemp*T
    return T2

#
# FUNCTION checktest
#
# Probabilistic acceptance criterion in simulated annealing
def checktest(Phi,Phistar,T):
    return Phistar < Phi or np.random.ranf()<math.exp((Phi-Phistar)/T) 


#
# SCORE FUNCTION (ENERGY)
#
# FUNCTION fitcomp
#
# Compute data fitness
def fitcomp(level,M,n,pairs,bi,bf,conf,matrix,p=1):
    fithic=0.
    fitrna=0.
    fitchip=0.
    for k in range(len(pairs)):
        if matrix[pairs[k]]==0.:
#            print('value of pair is zero', matrix[pairs[k]])
            matrix[pairs[k]]=1.
#        print(len(conf.bead))
        fithic=fithic+matrix[pairs[k]]*(M[pairs[k]]-conf.bead[pairs[k][0]].ext-conf.bead[pairs[k][1]].ext)**2
    fithic=fithic/LMAX                   # fitness normalized to LMAX
    
    if level==0:
        i=bi
        j=bi
        s=0
        pairsrna=[]
        pairschip=[]
        while s<bf:
            while j<bf and rnachip[j]==rnachip[i]:
                j=j+1
#                print(j)
            if rnachip[i]>0 and j-i>5:
                pairsrna.append((i,j-1))
            elif rnachip[i]<0 and j-i>5:
                pairschip.append((i,j-1))
            i=j
            s=i
            
        for k in range(len(pairsrna)):
            Mrna=M[int(pairsrna[k][0]-bi):int(pairsrna[k][1]-bi),int(pairsrna[k][0]-bi):int(pairsrna[k][1]-bi)]
            for t in range(-2,3):
                Mrna=Mrna-np.diag(np.diag(Mrna,t),t)
                for i in range(len(Mrna)):
                    for j in range(len(Mrna)):
                        if Mrna[i,j]==0:
                            Mrna[i,j]=LMAX*len(Mrna)
            fitrna=fitrna+(np.min(Mrna)-(LMIN+LMAX))**2
        fitrna=fitrna
        for k in range(len(pairschip)):
            Mchip=M[int(pairschip[k][0]-bi):int(pairschip[k][1]-bi),int(pairschip[k][0]-bi):int(pairschip[k][1]-bi)]
            fitchip=fitchip+(np.max(Mchip)-LMIN)**2
        fitchip=fitchip


##    if level==0:
##        i=bi
##        j=bi
##        s=0
##        pairsrna=[]
##        while s<bf:
###            print('s',s,len(rna[bi:bf]))
##            while j<bf and rna[j]==rna[i]:
##                j=j+1
###                print(j)
##            if rna[i]>0 and j-i>4:
##                pairsrna.append((i,j-1))
##            i=j
##            s=i
###        if bi==31:
###            print(pairsrna,len(pairsrna),fitrna)
###        print('len rnaseq=',len(pairsrna))
##        for k in range(len(pairsrna)):
####            print(pairsrna[k][0],pairsrna[k][1])
####            print(M[pairsrna[k][0]:pairsrna[k][1],pairsrna[k][0]:pairsrna[k][1]])
####            fitrna=fitrna+(np.max(M[int(pairsrna[k][0]-bi):int(pairsrna[k][1]-bi),int(pairsrna[k][0]-bi):int(pairsrna[k][1]-bi)])-LMAX*((pairsrna[k][1]-pairsrna[k][0])/2))**2
##            fitrna=fitrna+(np.min(M[int(pairsrna[k][0]-bi):int(pairsrna[k][1]-bi),int(pairsrna[k][0]-bi):int(pairsrna[k][1]-bi)])-LMAX*2)**2
##        fitrna=fitrna
###        print('fitrna=',fitrna)
##
##        i=bi
##        j=bi
##        s=0
##        pairschip=[]
##        while s<bf:
##            while j<bf and chip[j]==chip[i]:
##                j=j+1
###                print(j)
##            if chip[i]>0 and j-i>4:
##                pairschip.append((i,j-1))
##            i=j
##            s=i
###        print(pairschip)            
###        print('len chipseq=',len(pairschip))
##        for k in range(len(pairschip)):
##            fitchip=fitchip+(np.max(M[int(pairschip[k][0]-bi):int(pairschip[k][1]-bi),int(pairschip[k][0]-bi):int(pairschip[k][1]-bi)]))**2
##        fitchip=fitchip
###        print('fitchip=',fitchip)



    fit=fithic+fitrna+fitchip
#    fit=fithic
#        print(level,fit,fitrna,fitchip)                   
    if not p:
        print("data fithic   ",fithic)     # p = print(flag (default 1: no print)
        outdata.write("data fithic   "+str(fithic)+'\n')
        print("data fitrna   ",fitrna)     # p = print(flag (default 1: no print)
        outdata.write("data fitrna   "+str(fitrna)+'\n')
        print("data fitchip   ",fitchip)     # p = print(flag (default 1: no print)
        outdata.write("data fitchip   "+str(fitchip)+'\n')
        print("data fit   ",fit)     # p = print(flag (default 1: no print)
        outdata.write("data fit   "+str(fit)+'\n')

    return fit,fithic,fitrna,fitchip


# FUNCTION constrcomp
#
# Compute constraint part
def constrcomp(M,n,conf,p=1):
    constr=0.
    for i in range(n-1):
        for j in range(i+1,n):
            dmin=conf.bead[i].ext+conf.bead[j].ext
            constr=constr+(dmin/(2*M[i,j]))*(1.-(energy_scale*(M[i,j]-dmin))**energy_exp\
                                         /(1.+(energy_scale*np.abs(M[i,j]-dmin))**energy_exp))
    if not p:
        print("constraint ",constr)   # p = print(flag (default 1: no print))
        outdata.write("constraint "+str(constr)+'\n')
    return constr

# FUNCTION energy
#
# Sums the weighted constraint part to the data fit part
def energy(level,pairs,bi,bf,conf,matrix,lamb=0.,p=1):
    n=len(conf)
    M=distmatrix(conf)
    [fit,fithic,fitrna,fitchip]=fitcomp(level,M,n,pairs,bi,bf,conf,matrix,p=1)
    constr=constrcomp(M,n,conf,p=1)        # constr is the constraint part of the energy, 
                                            # not yet multiplied by parameter lamb
    constr=lamb*constr
    energ=fit+constr 
    return energ,fit,fithic,fitrna,fitchip,constr


#
# SIMULATED ANNEALING
#
# FUNCTION annealing
#
# Sets the regularization parameter, sets the start temperature, samples the energy distribution
def annealing(level,pairs,bi,bf,matrix,chain):
    n=len(matrix)
    conf=deepcopy(chain)
    conf.zerochain()
    m=len(conf)

    # Computes a regularization parameter from a fixed relative weight of the constraint part
    fitlist=[]
    constrlist=[]
    for j in range(avgenergy):
        for i in range(1,n):
            conflamb=conf.perturb(i)
            M=distmatrix(conflamb)
            [fitlamb,fithic,fitrna,fitchip]=fitcomp(level,M,m,pairs,bi,bf,conflamb,matrix)
            constrlamb=constrcomp(M,m,conflamb)
            conf=conflamb
            fitlist.append(fitlamb)
            constrlist.append(constrlamb)
    fitlist.sort()
    constrlist.sort()
    meanfit=np.mean(fitlist[0:int(-len(fitlist)//(100-percenergy))])
    meanconstr=np.mean(constrlist[0:int(-len(constrlist)//(100-percenergy))])
    lamb=meanfit/meanconstr*regulenergy
    print("lambda", lamb,'(average within '+str(int(percenergy))+'-th percentile)\n')
    outdata.write("lambda "+str(lamb)+'average within '+str(int(percenergy))+'-th percentile)\n')

#
#  Added to start from initial configuration (comment out if needed)
#
    conf=deepcopy(chain)
    conf.zerochain()
##
    [Phi,fit,fithic,fitrna,fitchip,constr]=energy(level,pairs,bi,bf,conf,matrix,lamb) # Initial energy

        
    # warm-up phase
    T=Tmax
    naccept=0
    iterat=0
    nincr=0
    nincracc=0
    accrate=0.
    while iterat < itwarm:
        iterat=iterat+1
        for i in range(1,n):
            confstar=conf.perturb(i)
            [Phistar,fitstar,fithic,fitrna,fitchip,constrstar]=energy(level,pairs,bi,bf,confstar,matrix,lamb)
            if Phistar > Phi: nincr=nincr+1
            if checktest(Phi,Phistar,T):
                naccept=naccept+1
                if Phistar > Phi: nincracc=nincracc+1
                conf=confstar
                Phi=Phistar
                fit=fitstar
                constr=constrstar
        try:
            accrate=1.*nincracc/nincr
        except:
            accrate='nincr=0'
        if iterat >= checkwarm and iterat%checkwarm==0 and type(accrate)==float:
            if accrate > muwarm:
                break
            else:
                print("T = ", T)
                outdata.write("T = "+str(T)+'\n')
                print("Acceptance rate",accrate)
                outdata.write("Acceptance rate "+str(accrate)+'\n')
                T=warm(T,accrate)
                naccept=0
                nincr=0
                nincracc=0
    string = "\nAnnealing - warm up\n# cycles "+str(iterat)\
             +"\nAcceptance rate "+str(accrate)\
             +"\nStart temperature "+str(T)+'\n\n'\
             +"Annealing - sampling\n"

    print(string,)
    outdata.write(string)

    # annealing phase
#
#  Added to start from initial configuration (uncomment if needed)
#
#    conf=deepcopy(chain)
    conf.zerochain()
    [Phi,fit,fithic,fitrna,fitchip,constr]=energy(level,pairs,bi,bf,conf,matrix,lamb) # Initial energy
    string = "\nStart energy "+str(Phi)\
             +"\nData fithic "+str(fithic)+"  Data fitrna "+str(fitrna)+"  Data fitchip "+str(fitchip)\
             +"\nData fit "+str(fit)+" Constraint "+str(constr)+'\n\n'
    print(string,)
    outdata.write(string)
##
    Tseries=[]
    Phiseries=[]
    Taccept=[]
    Phiaccept=[]
    iteraccept=[]
    naccept=0
    nincrease=0
    iterat=0
    while not checkstop(Phiseries,iterat): 
        iterat=iterat+1
        Tseries.append(T)
        Phiseries.append(Phi)
        for i in range(1,n):
            confstar=conf.perturb(i) # proposed update
            [Phistar,fit,fithic,fitrna,fitchip,constr]=energy(level,pairs,bi,bf,confstar,matrix,lamb,1)
            if checktest(Phi,Phistar,T):
                naccept=naccept+1
                Taccept.append(T)
                Phiaccept.append([fit,constr])
                iteraccept.append((iterat-1)*n+i+1)
                conf=confstar
                if Phistar > Phi:
                    nincrease=nincrease+1
                Phi=Phistar
        if not iterat%1000:  # print(values every 1000 cycles)
            if len(Phiaccept)>0:
                string = "T = "+str(T)\
                         +"\nenergy "+str(Phi)+" ("+str(Phiaccept[-1][0])+" + "+str(Phiaccept[-1][1])+")\n"
            else:
                string="T = "+str(T)+". No transition accepted in 1000 cycles."
            print(string,) 
            outdata.write(string)
        T=cool(T)

    string = "\nAnnealing - Final temperature "+str(T)\
             +"\nFinal energy "+str(Phi)\
             +"\nTotal # cycles "+str(iterat)\
             +"\nTotal accepted updates "+str(naccept)\
             +"\nUpdates increasing energy "+str(nincrease)+'\n'
    print(string,)
    outdata.write(string)

    Phiaccept=np.array(Phiaccept)
    return conf, Taccept, Phiaccept, iteraccept


#
# SIMULATED ANNEALING
#
# FUNCTION diffannealing
#
# Sets the regularization parameter, sets the start temperature, samples the energy distribution
def diffannealing(level,pairs,bi,bf,matrix,chain):
    T=900000
    decrtemp=0.9
    n=len(matrix)
    conf=deepcopy(chain)
    conf.zerochain()
    m=len(conf)

    # Computes a regularization parameter from a fixed relative weight of the constraint part
    fitlist=[]
    constrlist=[]
    for j in range(avgenergy):
        for i in range(1,n):
            conflamb=conf.perturb(i)
            M=distmatrix(conflamb)
            [fitlamb,fithic,fitrna,fitchip]=fitcomp(level,M,m,pairs,bi,bf,conflamb,matrix)
            constrlamb=constrcomp(M,m,conflamb)
            conf=conflamb
            fitlist.append(fitlamb)
            constrlist.append(constrlamb)
    fitlist.sort()
    constrlist.sort()
    meanfit=np.mean(fitlist[0:-len(fitlist)/(100-percenergy)])
    meanconstr=np.mean(constrlist[0:-len(constrlist)/(100-percenergy)])
    lamb=meanfit/meanconstr*regulenergy
    print("lambda", lamb,'(average within '+str(int(percenergy))+'-th percentile)\n')
    outdata.write("lambda "+str(lamb)+'average within '+str(int(percenergy))+'-th percentile)\n')

#
#  Added to start from initial configuration (comment out if needed)
#
    conf=deepcopy(chain)
    conf.zerochain()

    # annealing phase

    [Phi,fit,fithic,fitrna,fitchip,constr]=energy(level,pairs,bi,bf,conf,matrix,lamb) # Initial energy
    string = "\nStart energy "+str(Phi)\
             +"\nData fithic "+str(fithic)+"  Data fitrna "+str(fitrna)+"  Data fitchip "+str(fitchip)\
             +"\nData fit "+str(fit)+" Constraint "+str(constr)+'\n\n'
    print(string,)
    outdata.write(string)
##
    Tseries=[]
    Phiseries=[]
    Taccept=[]
    Phiaccept=[]
    iteraccept=[]
    naccept=0
    nincrease=0
    iterat=0
    while not checkstop(Phiseries,iterat): 
        iterat=iterat+1
        Tseries.append(T)
        Phiseries.append(Phi)
        for i in range(1,n):
            confstar=conf.perturb(i) # proposed update
            [Phistar,fit,fithic,fitrna,fitchip,constr]=energy(level,pairs,bi,bf,confstar,matrix,lamb,1)
            if checktest(Phi,Phistar,T):
                naccept=naccept+1
                Taccept.append(T)
                Phiaccept.append([fit,constr])
                iteraccept.append((iterat-1)*n+i+1)
                conf=confstar
                if Phistar > Phi:
                    nincrease=nincrease+1
                Phi=Phistar
        if not iterat%1000:  # print(values every 1000 cycles)
            if len(Phiaccept)>0:
                string = "T = "+str(T)\
                         +"\nenergy "+str(Phi)+" ("+str(Phiaccept[-1][0])+" + "+str(Phiaccept[-1][1])+")\n"
            else:
                string="T = "+str(T)+". No transition accepted in 1000 cycles."
            print(string,) 
            outdata.write(string)
        T=cool(T)

    string = "\nAnnealing - Final temperature "+str(T)\
             +"\nFinal energy "+str(Phi)\
             +"\nTotal # cycles "+str(iterat)\
             +"\nTotal accepted updates "+str(naccept)\
             +"\nUpdates increasing energy "+str(nincrease)+'\n'
    print(string,)
    outdata.write(string)

    Phiaccept=np.array(Phiaccept)
    return conf, Taccept, Phiaccept, iteraccept



# FUNCTION skipannealing, to be used in case of centromere of telomer
def skipannealing(pairs,matrix,chain):
    conf=deepcopy(chain)
    Taccept=[]
    Phiaccept=[]
    iteraccept=[]
    return conf, Taccept, Phiaccept, iteraccept


# FUNCTION binning
#
# Bins the contact matrix on the blocks identified at the current level
# Returns an upper triangular matrix
def binning(sizes,matrix):
    n=len(sizes)
    # zeroing lower triangle
    for i in range(1,len(matrix)):
        for j in range(i):
            matrix[i,j]=0
    binnedmatrix=np.zeros([n,n])
    for i in range(n):
        iinf=np.sum(sizes[0:i])
        for j in range(i,n):
            jinf=np.sum(sizes[0:j])
            binnedmatrix[i,j]=np.sum(matrix[int(iinf):int(iinf+sizes[i]),int(jinf):int(jinf+sizes[j])])
    return binnedmatrix
    




# FUNCTION compbead
#
# Computes a lower-resolution bead from a reconstructed subchain
def compbead(conf):
    bigbead=bead()
    # centroid
    bigbead.centrd[0]=sum(conf.bead[i].centrd[0] for i in range(len(conf)))/len(conf)
    bigbead.centrd[1]=sum(conf.bead[i].centrd[1] for i in range(len(conf)))/len(conf)
    bigbead.centrd[2]=sum(conf.bead[i].centrd[2] for i in range(len(conf)))/len(conf)
    # endpoints
    bigbead.endp1=conf.bead[0].endp1
    bigbead.endp2=conf.bead[-1].endp2
    # size
    L=np.matrix([conf.bead[i].centrd for i in range(len(conf))])
    M=L-bigbead.centrd
    T=M.transpose()*M
    [lambd,V]=LA.eigh(T)
    bigbead.ext=math.sqrt(abs(lambd).max())*extrate 
    c=bigbead.d1d2()
    d1=c[0]
    d2=c[1]
    ph=bigbead.innerangle()
    bigbead.endp1=np.array([0.,0.,0.])
    bigbead.centrd=np.array([d1,0.,0.])
    bigbead.endp2=np.array([d1-d2*math.cos(ph),d2*math.sin(ph),0.])
    return bigbead

    


# FUNCTION savepar
#
# Saves the output parameters of a specified configuration to file
def savepar(EHist,level,blockn):
    out=open(filen+'_'+timemark+'_'+str(level)+'_'+str(blockn)+'_Energy.txt',"w")
    for i in range(len(EHist)):
        out.write(str(EHist[i][0])+' '+str(EHist[i][1])+' '+str(EHist[i][0]+EHist[i][1])+'\n')
    out.close()
                  

# FUNCTION saveconf
#
# Saves a chain configuration to file
def saveconf(conf,level,blockn):
    if level==-1:
        M=distmatrix(conf)
        np.savetxt(filen+'_'+timemark+'_DistMat.txt',M)
        out=open(filen+'_'+timemark+'_LastConf.txt',"w")
        out.write(str(conf))
        out.close()
    else:
        out=open(filen+'_'+timemark+'_'+str(level)+'_'+str(blockn)+'.txt',"w")
        out.write(str(conf))
        out.close()

                


# FUNCTION constr
#
# Computes the bead sizes at level 0
def constr(n,chip,rna,NMAX):
    L=LMAX-(LMAX-LMIN)*n/NMAX
    if rna-chip==1.:
        L=L+0.1*L
    if rna-chip==-1.:
        L=L-0.1*L
    bead1=bead(np.array([0.,0.,0.]),np.array([L/2.,0.,0.]),np.array([L,0.,0.]),ext=L/2.) 
    return bead1

def centrsize(n):
    G=crate*RIS
    bead2=bead(np.array([0.,0.,0.]),np.array([G/2.,0.,0.]),np.array([G,0.,0.]),ext=G/2.)
    return bead2
    


# FUNCTION inizchain
#
# Returns the start configuration at level 0
def inizchain(hm,centromere,telomeres,chip,rna,NMAX=0):
    if not NMAX: NMAX=hm.max() # NMAX=0 means that the maximum frequency in the current data matrix is assumed
    chain1=chain([])
    for i in range(len(hm)):
        if len(centromere)==1 and i in range(centromere[0][0],centromere[0][1]+1):
            chain1.chainadd(centrsize(hm[i,i]))
        elif len(telomeres)==1 and i in range(telomeres[0][0],telomeres[0][1]+1):
            chain1.chainadd(centrsize(hm[i,i]))
        elif len(telomeres)==2 and (i in range(telomeres[0][0],telomeres[0][1]+1) or i in range(telomeres[1][0],telomeres[1][1]+1)):
            chain1.chainadd(centrsize(hm[i,i]))
        else:
            chain1.chainadd(constr(hm[i,i],chip[i],rna[i],NMAX)) 
    chain1.formchain()
    return chain1




# FUNCTION align
#
# Aligns a subchain to its low-resolution counterpart (rotated low-resolution bead at the successive level)
def align(cha,bd):
    ch=copy.deepcopy(cha)
    ch.zerochain() # Place the subchain in the origin
    # basis vectors for subchain
    v2=np.array([0.,0.,0.])
    v2[0]=sum(ch.bead[i].centrd[0] for i in range(len(ch)))*1./len(ch)
    v2[1]=sum(ch.bead[i].centrd[1] for i in range(len(ch)))*1./len(ch)
    v2[2]=sum(ch.bead[i].centrd[2] for i in range(len(ch)))*1./len(ch)
    v3=ch.bead[-1].endp2
    v1=np.cross(v2,v3)
    v1=np.array([v1[0],v1[1],v1[2]])
    if LA.norm(np.float128(v1))==0:
        v1[0]=v1[0]+0.0001
        v2[1]=v2[1]+0.0001
        v3[2]=v3[2]+0.0001
    base1=[v1/LA.norm(np.float128(v1)),v2/LA.norm(np.float128(v2)),v3/LA.norm(np.float128(v3))]

    bd.zerobead() # Place the low-resolution bead in the origin

    # basis vectors for bead
    w2=bd.centrd
    w3=bd.endp2
    w1=np.cross(w2,w3)
    w1=np.array([w1[0],w1[1],w1[2]])
    base2=[w1/LA.norm(np.float128(w1)),w2/LA.norm(np.float128(w2)),w3/LA.norm(np.float128(w3))]

    # transform base1 into base2 (Rotation matrix)
    M1=np.matrix(base1)
    M2=np.matrix(base2)
    M1inv=LA.inv(M1)
    Lin=M1inv*M2
    for j in range(len(ch)):
        ch.bead[j].zerobead()
        b=np.matrix(ch.bead[j].centrd)*Lin
        ch.bead[j].centrd=np.array([b[0,0],b[0,1],b[0,2]])
        b=np.matrix(ch.bead[j].endp2)*Lin
        ch.bead[j].endp2=np.array([b[0,0],b[0,1],b[0,2]])
    return ch
     




# FUNCTION compose
#
# Recursive procedure to reconstruct the full-resolution chain from the lower-resolution versions
def compose(conf,nlevel):
    for i in range(nlevel):
        level=nlevel-i-1
        if level>-1:
            Csuc=chain([])    # Initialize the higher-resolution chain
            for ibead in range(len(conf)):  # Scan the beads at the current resolution
                mat=np.loadtxt(filen+'_'+timemark+'_'+str(level)+'_'+str(ibead)+'.txt') # read partial data
                subchain=chain([])
                for i in range(int(len(mat)/3)):                # form the subchain for the current bead
                    subchain.chainadd(bead(np.array([mat[3*i,0],mat[3*i,1],mat[3*i,2]]),\
                                           np.array([mat[3*i+1,0],mat[3*i+1,1],mat[3*i+1,2]]),\
                                           np.array([mat[3*i+2,0],mat[3*i+2,1],mat[3*i+2,2]]),mat[3*i,3]))
                ch=align(subchain,conf.bead[ibead])       # Align the subchain with the current bead
                Csuc.chainmerge(ch)                        # Append the new subchain to the higher-resolution chain
            Csuc.formchain()                               # Form the higher-resolution chain
            conf=deepcopy(Csuc)

    return conf




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






# FUNCTION output
#
# Plots and saves the final configuration to file
def output(conf):
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax.set_aspect('equal')
    col_gen=cycle('bgrcmyk')
    xl=[]
    yl=[]
    zl=[]
    for i in range(len(conf)):
        x=conf.bead[i].centrd[0]
        y=conf.bead[i].centrd[1]
        z=conf.bead[i].centrd[2]
        xl.append(conf.bead[i].centrd[0])
        yl.append(conf.bead[i].centrd[1])
        zl.append(conf.bead[i].centrd[2])
        r=conf.bead[i].ext

## Display spherical beads (uncomment if needed)
##        (xs,ys,zs) = drawSphere(x,y,z,r)
##        ax.plot_wireframe(xs, ys, zs, color=col_gen.next())    

    ax.plot(xl, yl, zl,'r',linewidth=3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    #set_axes_equal(ax)
    plt.show()





# FUNCTION chromstruct
#
# Recursive procedure to reconstruct the full-resolution chain
def chromstruct(contacts,chaininf,bigblocks,level=0):
    nblocks=[len(contacts)]
    nlevel=0
    while nblocks[nlevel]>1:
        [blocks,sizes,bounds]=block(nlevel,contacts,minsize=minsize,window=window,span=span) # extracts the TADs at the current level
        nblocks[nlevel]=len(blocks)
        nblocks.append(nblocks[nlevel])
        out=open(filen+'_'+timemark+'_'+str(nlevel)+'_BlockSizes.txt',"w")
        for i in range(nblocks[nlevel]):
            out.write(str(sizes[i])+'\n')
        out.close()

        string=       "\n\n========================================\nCHROMSTRUCT ver "+version\
                      +"\n========================================"\
                      +"\n level "+str(nlevel)\
                      +"\n# Blocks "+str(nblocks[nlevel])\
                      +"\n(smoothing window "+str(window)+"; averaging span "+str(span)+"; minimum block size "+str(minsize)+")"\
                      +"\n# Neglected diagonals "+str(diagneg)\
                      +"\nEnergy: relative weight, constraint part: "+str(regulenergy)\
                      +"\nEnergy: neighbor interaction function: scale = "+str(energy_scale)+"; exponent = "+str(energy_exp)\
                      +"\nAssumed bead sizes: max sigma * "+str(extrate)\
                      +"\nStop energy relative tolerance: "+str(StopTolerance)+" over "+str(itstop)+" cycles\n"
        print(string,)
        outdata.write(string)
       
        newchain=chain([])

        chainlist=[]
        
        for i in range(1,nblocks[nlevel]+1):         # Form a list of subchains at the current level
            chainlist.append(chain(chaininf.bead[bounds[i-1]:bounds[i]]))
        
        for i in range(nblocks[nlevel]):             # Scan the subchains (intrinsically parallelizable)
            string = "\n\n========================================\nlevel "+str(nlevel)+" block "+str(i)+'\n'
            print(string,)
            outdata.write(string)     
            
            L=populate(blocks[i]) # select the pairs for the data fit part
            string = "Relevant pairs "+str(L)+'\n\n'
            print(string,)
            outdata.write(string)

##            if nlevel>0:
##                print('dati per annealing')
##                print(L)
##                print(blocks[i]
##                print(chainlist[i])
            
            if blocks[i].max()==-1:
                [C, Tseries, Phiseries, iteraccept] = skipannealing(L,blocks[i],chainlist[i]) # Sample the current subchain
            elif len(blocks[i])>40:
                [C, Tseries, Phiseries, iteraccept] = diffannealing(nlevel,L,bounds[i],bounds[i+1]-1,blocks[i],chainlist[i]) # Sample the current subchain
            else:
                print('bi=',bounds[i],'bf=',bounds[i+1])
                [C, Tseries, Phiseries, iteraccept] = annealing(nlevel,L,bounds[i],bounds[i+1]-1,blocks[i],chainlist[i]) # Sample the current subchain

            #
            # (Optional) Plot data fit and constraint in the current cycle
            #

            if display[0] == 'y':
                output(C)
##                pylab.plot(Phiseries[:,0],label='Data score')
##                pylab.plot(Phiseries[:,1],label='Constraint score')
##                pylab.plot(Phiseries[:,0]+Phiseries[:,1],label='Total score')
##                plt.xlabel("Accepted updates")
##                pylab.legend()
##                pylab.show()
##
##                # Plot accepted vs proposed updates in the current cycle
##                pylab.plot(iteraccept,range(len(iteraccept)))
##                plt.ylabel('Accepted update')
##                plt.xlabel('Proposed update')
##                pylab.show()

            # end (Optional)
            #
            
            savepar(Phiseries, nlevel, i)                  # Save the features of the annealing schedule
            saveconf(C,nlevel,i)                           # Save current subchain configuration
            newbead=compbead(C)                           # Compute lower-resolution bead from current subchain
            newchain.chainadd(deepcopy(newbead))     # Append lower-res bead to lower-res chain
        newchain.formchain()
        chaininf=newchain
        contacts=binning(sizes,contacts)              # Bin data matrix to successive resolution

        nlevel=nlevel+1                             # Increase level if appropriate
    nlevel=nlevel-1
    if nlevel>0:    
        Clast=compose(C,nlevel)
    else:
        Clast=C
    
    string = "CPU time (secs) "+str(time.process_time())+'\n'
    print(string,)
    outdata.write(string)
        
    saveconf(Clast,-1,-1)
    output(Clast)                                 # Plot and save final result
    return





#===================================================================================================
#
# MAIN
#
#         POSSIBLY COMPLETE BY READING THE INITIAL PARAMETERS FROM EXTERNAL FILE OR TERMINAL
#         (SEE TOP OF FILE FOR THE DEFAULT VALUES)


### PARAMETERS
###==================================================================================================
###
##
###
###
### PENALTY FUNCTION (ENERGY) TUNING
###
### Fitness part
##diagneg=              # # neglected diagonals in contact matrix (in populate, to select the relevant pairs for data fit)
##datafact=             # Fraction of the pairs per block used to build the penalty function (real, in populate)
##
### Constraint part
##energy_scale=              # Tunes the extent of the moderate penalty range around the threshold distance 
##                           # (floating; physical dimensions 1/length; measurement unit: reciprocal of the unit used for DIA)
##energy_exp=                # Tunes the slopes of the penalty in the transitions between the moderate range
##                           # and the external intervals (odd integer, dimensionless)
###
###
### TAD EXTRACTION (FUNCTION block)
##span=                 # SIZE OF THE MOVING TRIANGLE
##window=               # MOVING AVERAGE WINDOW APERTURE
##minsize=              # MINIMUM REQUIRED DIAGONAL BLOCK SIZE
##
##
##
###
### ANNEALING: DETERMINATION OF THE REGULARIZATION PARAMETER lamb
##regulenergy=          # TARGET RELATIVE WEIGHT CONSTRAINT/FITNESS
##avgenergy=            # FIXED NUMBER OF AVERAGING CYCLES TO EVALUATE THE CONSTRAINT/FITNESS RATIO
##percenergy=           # MAXIMUM ACCEPTED PERCENTILE IN THE ENERGY SAMPLE SETS (integer <= 100)
##
###
### ANNEALING: WARM-UP PHASE
##Tmax=                 # Start temperature, warm-up phase
##itwarm=               # Max # warm-up cycles
##incrtemp=             # Fixed temperature increase coefficient at warm-up
##checkwarm=            # Fixed milestone on warm-up cycles to check the acceptance rate
##muwarm=               # Minimum acceptance rate to end warm-up
##
###
### ANNEALING: SAMPLING PHASE
##itmax=                # Max # annealing cycles (positive integer)
##itstop=               # # consecutive cycles with energy variations within StopTolerance to stop annealing
##StopTolerance=        # Stop tolerance
##RANDPLA=2*            # Max planar angle perturbation at each update (floating, radians, doubled for convenience)
##RANDDIE=2*            # Max dihedral angle perturbation at each update (floating, radians, doubled for convenience)
##decrtemp=             # Fixed cooling coefficient at each annealing cicle
##
##
###
### GEOMETRIC PARAMETERS
##DIA=                  # Chromatin fiber diameter (nm)
##RIS=                  # Contact matrix (full) genomic resolution (kbp)
##NB=                   # kilobase-pairs in a chromatin fragment with length DIA
##crate=                # compactness rate for telomeres and centromeres
##
##    
##NC=RIS/NB             # # fragments per genomic resolution unit
##LMAX=math.sqrt(RIS)*DIA/NB*np.pi     # hypothesized maximum size of a bead in the full-resolution model (nm)
##LMIN=NC**(1./3)*math.sqrt(3)*DIA # hypothesized minimum bead size in the full-resolution model (nm)
##NMAX=                 # Maximum contact frequency: fixed or computed in the current data matrix (see function inizchain)
##extrate=              # Size tuning for beads at levels > 0
##
###==================================================================================================
###

string=       "\n\n========================================\nCHROMSTRUCT ver "+version\
              +"\nCopyright (C) 2020 Emanuele Salerno, Claudia Caudai\n\n"\
              +"This program comes with ABSOLUTELY NO WARRANTY; for details type `sw'.\n"\
              +"This is free software, and you are welcome to redistribute it\n"\
              +"under certain conditions; type `sc' for details.\n"
print(string)
opt=input("type 'sw' for warranty, 'sc' for conditions, or <enter> to continue\n\n")
if opt=="sw":
    print(warranty)
    opt2=input("type 'sc' for conditions, or <enter> to continue\n\n")
    if opt2=="sc": print(conditions)
elif opt=="sc":
    print(conditions)
    opt2=input("type 'sw' for warranty, or <enter> to continue\n\n")
    if opt2=="sw": print(warranty)


# timestamp to identify all the partial result files in the current run
t=time.localtime()
timemark=str(t.tm_year)[2:4]+"-%02d-%02d-%02d%02d" % (t.tm_mon,t.tm_mday,t.tm_hour,t.tm_min)



filen=input('File name (.txt) = ? ')      # Read data file name

try: hm=np.loadtxt(filen+'.txt')              # Read contact data
except:
    try: hm=np.loadtxt(filen)
    except:
        print('No such file: no data read\n')
        sys.exit(0)

rna=np.zeros(len(hm))                       
chip=np.zeros(len(hm))
rnachip=np.zeros(len(hm))
ctcf=np.zeros((len(hm),len(hm)))
rn=[]
ch=[]


if filen[-4:len(filen)]=='.txt':
    filen=filen[0:-4]

filer=input('RNAseq file(.txt) = ? ')      # Read RNAseq file name

try: rn=np.loadtxt(filer+'.txt')              # Read RNAseq data
except:
    try: rn=np.loadtxt(filer)
    except:
        print('No such file: no data read\n')
#        sys.exit(0)


filec=input('ChIPseq file(.txt) = ? ')      # Read ChIPseq file name

try: ch=np.loadtxt(filec+'.txt')              # Read ChIPseq data
except:
    try: ch=np.loadtxt(filec)
    except:
        print('No such file: no data read\n')
#        sys.exit(0)


filect=input('CTCF file(.txt) = ? ')      # Read CTCF file name

try: ctcf=np.loadtxt(filect+'.txt')              # Read ChIPseq data
except:
    try: ch=np.loadtxt(filec)
    except:
        print('No such file: no data read\n')
#        sys.exit(0)
if len(ctcf)==len(hm):
    for i in range(len(hm)):
        for j in range(len(hm)):
            hm[i,j]=hm[i,j]+ctcf[i,j]*100
else:
    print('Wrong CTCF format file')
        
pairsrna=[]
if len(rn)==len(rna):
    rna=rn
else:
    print('Wrong RNAseq format file')


pairschip=[]                          
if len(ch)==len(chip):
    chip=ch
else:
    print('Wrong ChIPseq format file')

for i in range(len(hm)):
    rnachip[i]=rna[i]-chip[i]

outdata=open(filen+'_'+timemark+'_Log.txt',"w")  # Open logfile

if NMAX:
    string="\n\nFIXED NMAX = "+str(NMAX)
    print(string)
    outdata.write(string)

[centromere, telomeres, bigblocks]=centr(hm)                          # Find centromere if exists
print('centromere', centromere)
np.savetxt(filen+'_'+timemark+'_centromere.txt',centromere)                         # Find telomeres if exist
print('telomeres', telomeres)
np.savetxt(filen+'_'+timemark+'_telomeres.txt',telomeres)

conf=inizchain(hm,centromere,telomeres,chip,rna,NMAX)     # Set initial guess


for i in range(len(hm)):                         # Set the lower triangle of the contact matrix to zero
    for j in range(i):
        hm[i,j]=0.

display=input("Do you need intermediate plots? (y/n) ")
if display=='':display='n'

chromstruct(hm,conf,bigblocks)                             # Call iteration

outdata.close()                                  # Close logfile


