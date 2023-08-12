# ChromStruct v 3.1

# Reconstruction of the 3D structure of the chromatin fiber from Hi-C
# contact frequency data.

# Copyright (C) 2016 Emanuele Salerno, Claudia Caudai (ISTI-CNR Pisa)
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
          +"GNU General Public License version 3 for more details"\
          +"<http://www.gnu.org/licenses/>.\n\n"

conditions="This program is free software: you can redistribute it and/or modify\n"\
            +"it under the terms of the GNU General Public License as published by\n"\
            +"the Free Software Foundation, version 3 of the License"\
            +"<http://www.gnu.org/licenses/>.\n\n"



##You should have received a copy of the GNU General Public License
##along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Date 06/07/2016

# References:
# Caudai C., Salerno E., Zoppè M., Tonazzini A.
# Inferring 3D chromatin structure using a multiscale
# approach based on quaternions.
# In: BMC Bioinformatics, vol. 16, article n. 234. BioMed Central, 2015.
# DOI: 10.1186/s12859-015-0667-0
#
# Caudai C., Salerno E., Zoppè M., Tonazzini, A.
# 3D chromatin structure estimation through a constraint-enhanced score function.
# BMC Bioinformatics, 2016, to appear
#


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


import copy
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

# PARAMETERS
#==================================================================================================
#

#
# CODE VERSION
version="3.1 (June 7, 2016)"

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
span=7                # SIZE OF THE MOVING TRIANGLE
window=3              # MOVING AVERAGE WINDOW APERTURE
minsize=7             # MINIMUM REQUIRED DIAGONAL BLOCK SIZE


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
itstop=500            # # consecutive cycles with energy variations within StopTolerance to stop annealing
StopTolerance=1.e-5   # Stop tolerance
RANDPLA=2*0.05        # Max planar angle perturbation at each update (floating, radians, doubled for convenience)
RANDDIE=2*0.05        # Max dihedral angle perturbation at each update (floating, radians, doubled for convenience)
decrtemp=0.998        # Fixed cooling coefficient at each annealing cicle


#
# GEOMETRIC PARAMETERS
DIA=30.               # Chromatin fiber diameter (nm)
RIS=100.              # Contact matrix (full) genomic resolution (kbp)
NB=3.                 # kilobase-pairs in a chromatin fragment with length DIA

    
NC=RIS/NB             # # fragments per genomic resolution unit
LMAX=NC*DIA/np.pi     # hypothesized maximum size of a bead in the full-resolution model (nm)
LMIN=NC**(1./3)*math.sqrt(3)*DIA # hypothesized minimum bead size in the full-resolution model (nm)
NMAX=0.               # Maximum contact frequency: fixed or computed in the current data matrix (see function inizchain)
extrate=0.3           # Size tuning for beads at levels > 0

#==================================================================================================
#


#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# DEFINITIONS (STANDARD)

# Definition of 2D and 3D vector,
# matrix, quaternion and geometry module (euclid3-0.01)
# downloaded from https://pypi.python.org/pypi/euclid3/0.01


__docformat__ = 'restructuredtext'
__version__ = '$Id: euclid.py 37 2011-08-21 22:24:05Z elfnor@gmail.com $'
__revision__ = '$Revision: 37 $'


import math 
import operator
import types

# Some magic here.  If _use_slots is True, the classes will derive from
# object and will define a __slots__ class variable.  If _use_slots is
# False, classes will be old-style and will not define __slots__.
#
# _use_slots = True:   Memory efficient, probably faster in future versions
#                      of Python, "better".
# _use_slots = False:  Ordinary classes, much faster than slots in current
#                      versions of Python (2.4 and 2.5).
_use_slots = True

# If True, allows components of Vector2 and Vector3 to be set via swizzling;
# e.g.  v.xyz = (1, 2, 3).  This is much, much slower than the more verbose
# v.x = 1; v.y = 2; v.z = 3,  and slows down ordinary element setting as
# well.  Recommended setting is False.
_enable_swizzle_set = False

# Requires class to derive from object.
if _enable_swizzle_set:
    _use_slots = True

# Implement _use_slots magic.
class _EuclidMetaclass(type):
    def __new__(cls, name, bases, dct):
        if '__slots__' in dct:
            dct['__getstate__'] = cls._create_getstate(dct['__slots__'])
            dct['__setstate__'] = cls._create_setstate(dct['__slots__'])
        if _use_slots:
            return type.__new__(cls, name, bases + (object,), dct)
        else:
            if '__slots__' in dct:
                del dct['__slots__']
            return types.ClassType.__new__(types.ClassType, name, bases, dct)

    @classmethod
    def _create_getstate(cls, slots):
        def __getstate__(self):
            d = {}
            for slot in slots:
                d[slot] = getattr(self, slot)
            return d
        return __getstate__

    @classmethod
    def _create_setstate(cls, slots):
        def __setstate__(self, state):
            for name, value in state.items():
                setattr(self, name, value)
        return __setstate__

__metaclass__ = _EuclidMetaclass

class Vector2:
    __slots__ = ['x', 'y']
    __hash__ = None

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __copy__(self):
        return self.__class__(self.x, self.y)

    copy = __copy__

    def __repr__(self):
        return 'Vector2(%.2f, %.2f)' % (self.x, self.y)

    def __eq__(self, other):
        if isinstance(other, Vector2):
            return self.x == other.x and \
                   self.y == other.y
        else:
            assert hasattr(other, '__len__') and len(other) == 2
            return self.x == other[0] and \
                   self.y == other[1]

    def __ne__(self, other):
        return not self.__eq__(other)

    def __nonzero__(self):
        return self.x != 0 or self.y != 0

    def __len__(self):
        return 2

    def __getitem__(self, key):
        return (self.x, self.y)[key]

    def __setitem__(self, key, value):
        l = [self.x, self.y]
        l[key] = value
        self.x, self.y = l

    def __iter__(self):
        return iter((self.x, self.y))

    def __getattr__(self, name):
        try:
            return tuple([(self.x, self.y)['xy'.index(c)] \
                          for c in name])
        except ValueError:
            raise AttributeError(name)

    if _enable_swizzle_set:
        # This has detrimental performance on ordinary setattr as well
        # if enabled
        def __setattr__(self, name, value):
            if len(name) == 1:
                object.__setattr__(self, name, value)
            else:
                try:
                    l = [self.x, self.y]
                    for c, v in map(None, name, value):
                        l['xy'.index(c)] = v
                    self.x, self.y = l
                except ValueError:
                    raise AttributeError(name)

    def __add__(self, other):
        if isinstance(other, Vector2):
            # Vector + Vector -> Vector
            # Vector + Point -> Point
            # Point + Point -> Vector
            if self.__class__ is other.__class__:
                _class = Vector2
            else:
                _class = Point2
            return _class(self.x + other.x,
                          self.y + other.y)
        else:
            assert hasattr(other, '__len__') and len(other) == 2
            return Vector2(self.x + other[0],
                           self.y + other[1])
    __radd__ = __add__

    def __iadd__(self, other):
        if isinstance(other, Vector2):
            self.x += other.x
            self.y += other.y
        else:
            self.x += other[0]
            self.y += other[1]
        return self

    def __sub__(self, other):
        if isinstance(other, Vector2):
            # Vector - Vector -> Vector
            # Vector - Point -> Point
            # Point - Point -> Vector
            if self.__class__ is other.__class__:
                _class = Vector2
            else:
                _class = Point2
            return _class(self.x - other.x,
                          self.y - other.y)
        else:
            assert hasattr(other, '__len__') and len(other) == 2
            return Vector2(self.x - other[0],
                           self.y - other[1])

   
    def __rsub__(self, other):
        if isinstance(other, Vector2):
            return Vector2(other.x - self.x,
                           other.y - self.y)
        else:
            assert hasattr(other, '__len__') and len(other) == 2
            return Vector2(other.x - self[0],
                           other.y - self[1])

    def __mul__(self, other):
        assert type(other) in (int, int, float)
        return Vector2(self.x * other,
                       self.y * other)

    __rmul__ = __mul__

    def __imul__(self, other):
        assert type(other) in (int, int, float)
        self.x *= other
        self.y *= other
        return self

    def __div__(self, other):
        assert type(other) in (int, int, float)
        return Vector2(operator.div(self.x, other),
                       operator.div(self.y, other))


    def __rdiv__(self, other):
        assert type(other) in (int, int, float)
        return Vector2(operator.div(other, self.x),
                       operator.div(other, self.y))

    def __floordiv__(self, other):
        assert type(other) in (int, int, float)
        return Vector2(operator.floordiv(self.x, other),
                       operator.floordiv(self.y, other))


    def __rfloordiv__(self, other):
        assert type(other) in (int, int, float)
        return Vector2(operator.floordiv(other, self.x),
                       operator.floordiv(other, self.y))

    def __truediv__(self, other):
        assert type(other) in (int, int, float)
        return Vector2(operator.truediv(self.x, other),
                       operator.truediv(self.y, other))


    def __rtruediv__(self, other):
        assert type(other) in (int, int, float)
        return Vector2(operator.truediv(other, self.x),
                       operator.truediv(other, self.y))
    
    def __neg__(self):
        return Vector2(-self.x,
                        -self.y)

    __pos__ = __copy__
    
    def __abs__(self):
        return math.sqrt(self.x ** 2 + \
                         self.y ** 2)

    magnitude = __abs__

    def magnitude_squared(self):
        return self.x ** 2 + \
               self.y ** 2

    def normalize(self):
        d = self.magnitude()
        if d:
            self.x /= d
            self.y /= d
        return self

    def normalized(self):
        d = self.magnitude()
        if d:
            return Vector2(self.x / d, 
                           self.y / d)
        return self.copy()

    def dot(self, other):
        assert isinstance(other, Vector2)
        return self.x * other.x + \
               self.y * other.y

    def cross(self):
        return Vector2(self.y, -self.x)

    def reflect(self, normal):
        # assume normal is normalized
        assert isinstance(normal, Vector2)
        d = 2 * (self.x * normal.x + self.y * normal.y)
        return Vector2(self.x - d * normal.x,
                       self.y - d * normal.y)

    def angle(self, other):
        """Return the angle to the vector other"""
        return math.acos(self.dot(other) / (self.magnitude()*other.magnitude()))

    def project(self, other):
        """Return one vector projected on the vector other"""
        n = other.normalized()
        return self.dot(n)*n

class Vector3:
    __slots__ = ['x', 'y', 'z']
    __hash__ = None

    def __init__(self, x=0, y=0, z=0):
        self.x = np.float128(x)    # MODIFIED FOR PRECISION!!
        self.y = np.float128(y)
        self.z = np.float128(z)
##        self.x = x
##        self.y = y
##        self.z = z


    def __copy__(self):
        return self.__class__(self.x, self.y, self.z)

    copy = __copy__

    def __repr__(self):
        return 'Vector3(%.2f, %.2f, %.2f)' % (self.x,
                                              self.y,
                                              self.z)

    def __eq__(self, other):
        if isinstance(other, Vector3):
            return self.x == other.x and \
                   self.y == other.y and \
                   self.z == other.z
        else:
            assert hasattr(other, '__len__') and len(other) == 3
            return self.x == other[0] and \
                   self.y == other[1] and \
                   self.z == other[2]

    def __ne__(self, other):
        return not self.__eq__(other)

    def __nonzero__(self):
        return self.x != 0 or self.y != 0 or self.z != 0

    def __len__(self):
        return 3

    def __getitem__(self, key):
        return (self.x, self.y, self.z)[key]

    def __setitem__(self, key, value):
        l = [self.x, self.y, self.z]
        l[key] = value
        self.x, self.y, self.z = l

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    def __getattr__(self, name):
        try:
            return tuple([(self.x, self.y, self.z)['xyz'.index(c)] \
                          for c in name])
        except ValueError:
            raise AttributeError(name)

    if _enable_swizzle_set:
        # This has detrimental performance on ordinary setattr as well
        # if enabled
        def __setattr__(self, name, value):
            if len(name) == 1:
                object.__setattr__(self, name, value)
            else:
                try:
                    l = [self.x, self.y, self.z]
                    for c, v in map(None, name, value):
                        l['xyz'.index(c)] = v
                    self.x, self.y, self.z = l
                except ValueError:
                    raise AttributeError(name)


    def __add__(self, other):
        if isinstance(other, Vector3):
            # Vector + Vector -> Vector
            # Vector + Point -> Point
            # Point + Point -> Vector
            if self.__class__ is other.__class__:
                _class = Vector3
            else:
                _class = Point3
            return _class(self.x + other.x,
                          self.y + other.y,
                          self.z + other.z)
        else:
            assert hasattr(other, '__len__') and len(other) == 3
            return Vector3(self.x + other[0],
                           self.y + other[1],
                           self.z + other[2])
    __radd__ = __add__

    def __iadd__(self, other):
        if isinstance(other, Vector3):
            self.x += other.x
            self.y += other.y
            self.z += other.z
        else:
            self.x += other[0]
            self.y += other[1]
            self.z += other[2]
        return self

    def __sub__(self, other):
        if isinstance(other, Vector3):
            # Vector - Vector -> Vector
            # Vector - Point -> Point
            # Point - Point -> Vector
            if self.__class__ is other.__class__:
                _class = Vector3
            else:
                _class = Point3
            return Vector3(self.x - other.x,
                           self.y - other.y,
                           self.z - other.z)
        else:
            assert hasattr(other, '__len__') and len(other) == 3
            return Vector3(self.x - other[0],
                           self.y - other[1],
                           self.z - other[2])

   
    def __rsub__(self, other):
        if isinstance(other, Vector3):
            return Vector3(other.x - self.x,
                           other.y - self.y,
                           other.z - self.z)
        else:
            assert hasattr(other, '__len__') and len(other) == 3
            return Vector3(other.x - self[0],
                           other.y - self[1],
                           other.z - self[2])

    def __mul__(self, other):
        if isinstance(other, Vector3):
            # TODO component-wise mul/div in-place and on Vector2; docs.
            if self.__class__ is Point3 or other.__class__ is Point3:
                _class = Point3
            else:
                _class = Vector3
            return _class(self.x * other.x,
                          self.y * other.y,
                          self.z * other.z)
        else: 
            assert type(other) in (int, int, float)
            return Vector3(self.x * other,
                           self.y * other,
                           self.z * other)

    __rmul__ = __mul__

    def __imul__(self, other):
        assert type(other) in (int, int, float)
        self.x *= other
        self.y *= other
        self.z *= other
        return self

    def __div__(self, other):
        assert type(other) in (int, int, float)
        return Vector3(operator.div(self.x, other),
                       operator.div(self.y, other),
                       operator.div(self.z, other))


    def __rdiv__(self, other):
        assert type(other) in (int, int, float)
        return Vector3(operator.div(other, self.x),
                       operator.div(other, self.y),
                       operator.div(other, self.z))

    def __floordiv__(self, other):
        assert type(other) in (int, int, float)
        return Vector3(operator.floordiv(self.x, other),
                       operator.floordiv(self.y, other),
                       operator.floordiv(self.z, other))


    def __rfloordiv__(self, other):
        assert type(other) in (int, int, float)
        return Vector3(operator.floordiv(other, self.x),
                       operator.floordiv(other, self.y),
                       operator.floordiv(other, self.z))

    def __truediv__(self, other):
        assert type(other) in (int, int, float)
        return Vector3(operator.truediv(self.x, other),
                       operator.truediv(self.y, other),
                       operator.truediv(self.z, other))


    def __rtruediv__(self, other):
        assert type(other) in (int, int, float)
        return Vector3(operator.truediv(other, self.x),
                       operator.truediv(other, self.y),
                       operator.truediv(other, self.z))
    
    def __neg__(self):
        return Vector3(-self.x,
                        -self.y,
                        -self.z)

    __pos__ = __copy__
    
    def __abs__(self):
        return math.sqrt(self.x ** 2 + \
                         self.y ** 2 + \
                         self.z ** 2)

    magnitude = __abs__

    def magnitude_squared(self):
        return self.x ** 2 + \
               self.y ** 2 + \
               self.z ** 2

    def normalize(self):
        d = self.magnitude()
        if d:
            self.x /= d
            self.y /= d
            self.z /= d
        return self

    def normalized(self):
        d = self.magnitude()
        if d:
            return Vector3(self.x / d, 
                           self.y / d, 
                           self.z / d)
        return self.copy()

    def dot(self, other):
        assert isinstance(other, Vector3)
        return self.x * other.x + \
               self.y * other.y + \
               self.z * other.z

    def cross(self, other):
        assert isinstance(other, Vector3)
        return Vector3(self.y * other.z - self.z * other.y,
                       -self.x * other.z + self.z * other.x,
                       self.x * other.y - self.y * other.x)

    def reflect(self, normal):
        # assume normal is normalized
        assert isinstance(normal, Vector3)
        d = 2 * (self.x * normal.x + self.y * normal.y + self.z * normal.z)
        return Vector3(self.x - d * normal.x,
                       self.y - d * normal.y,
                       self.z - d * normal.z)

    def rotate_around(self, axis, theta):
        """Return the vector rotated around axis through angle theta. Right hand rule applies"""

        # Adapted from equations published by Glenn Murray.
        # http://inside.mines.edu/~gmurray/ArbitraryAxisRotation/ArbitraryAxisRotation.html
        x, y, z = self.x, self.y,self.z
        u, v, w = axis.x, axis.y, axis.z

        # Extracted common factors for simplicity and efficiency
        r2 = u**2 + v**2 + w**2
        r = math.sqrt(r2)
        ct = math.cos(theta)
        st = math.sin(theta) / r
        dt = (u*x + v*y + w*z) * (1 - ct) / r2
        return Vector3((u * dt + x * ct + (-w * y + v * z) * st),
                       (v * dt + y * ct + ( w * x - u * z) * st),
                       (w * dt + z * ct + (-v * x + u * y) * st))

    def angle(self, other):
        """Return the angle to the vector other"""
        return math.acos(self.dot(other) / (self.magnitude()*other.magnitude()))

    def project(self, other):
        """Return one vector projected on the vector other"""
        n = other.normalized()
        return self.dot(n)*n

# a b c 
# e f g 
# i j k 

class Matrix3:
    __slots__ = list('abcefgijk')

    def __init__(self):
        self.identity()

    def __copy__(self):
        M = Matrix3()
        M.a = self.a
        M.b = self.b
        M.c = self.c
        M.e = self.e 
        M.f = self.f
        M.g = self.g
        M.i = self.i
        M.j = self.j
        M.k = self.k
        return M

    copy = __copy__
    def __repr__(self):
        return ('Matrix3([% 8.2f % 8.2f % 8.2f\n'  \
                '         % 8.2f % 8.2f % 8.2f\n'  \
                '         % 8.2f % 8.2f % 8.2f])') \
                % (self.a, self.b, self.c,
                   self.e, self.f, self.g,
                   self.i, self.j, self.k)

    def __getitem__(self, key):
        return [self.a, self.e, self.i,
                self.b, self.f, self.j,
                self.c, self.g, self.k][key]

    def __setitem__(self, key, value):
        L = self[:]
        L[key] = value
        (self.a, self.e, self.i,
         self.b, self.f, self.j,
         self.c, self.g, self.k) = L

    def __mul__(self, other):
        if isinstance(other, Matrix3):
            # Caching repeatedly accessed attributes in local variables
            # apparently increases performance by 20%.  Attrib: Will McGugan.
            Aa = self.a
            Ab = self.b
            Ac = self.c
            Ae = self.e
            Af = self.f
            Ag = self.g
            Ai = self.i
            Aj = self.j
            Ak = self.k
            Ba = other.a
            Bb = other.b
            Bc = other.c
            Be = other.e
            Bf = other.f
            Bg = other.g
            Bi = other.i
            Bj = other.j
            Bk = other.k
            C = Matrix3()
            C.a = Aa * Ba + Ab * Be + Ac * Bi
            C.b = Aa * Bb + Ab * Bf + Ac * Bj
            C.c = Aa * Bc + Ab * Bg + Ac * Bk
            C.e = Ae * Ba + Af * Be + Ag * Bi
            C.f = Ae * Bb + Af * Bf + Ag * Bj
            C.g = Ae * Bc + Af * Bg + Ag * Bk
            C.i = Ai * Ba + Aj * Be + Ak * Bi
            C.j = Ai * Bb + Aj * Bf + Ak * Bj
            C.k = Ai * Bc + Aj * Bg + Ak * Bk
            return C
        elif isinstance(other, Point2):
            A = self
            B = other
            P = Point2(0, 0)
            P.x = A.a * B.x + A.b * B.y + A.c
            P.y = A.e * B.x + A.f * B.y + A.g
            return P
        elif isinstance(other, Vector2):
            A = self
            B = other
            V = Vector2(0, 0)
            V.x = A.a * B.x + A.b * B.y 
            V.y = A.e * B.x + A.f * B.y 
            return V
        else:
            other = other.copy()
            other._apply_transform(self)
            return other

    def __imul__(self, other):
        assert isinstance(other, Matrix3)
        # Cache attributes in local vars (see Matrix3.__mul__).
        Aa = self.a
        Ab = self.b
        Ac = self.c
        Ae = self.e
        Af = self.f
        Ag = self.g
        Ai = self.i
        Aj = self.j
        Ak = self.k
        Ba = other.a
        Bb = other.b
        Bc = other.c
        Be = other.e
        Bf = other.f
        Bg = other.g
        Bi = other.i
        Bj = other.j
        Bk = other.k
        self.a = Aa * Ba + Ab * Be + Ac * Bi
        self.b = Aa * Bb + Ab * Bf + Ac * Bj
        self.c = Aa * Bc + Ab * Bg + Ac * Bk
        self.e = Ae * Ba + Af * Be + Ag * Bi
        self.f = Ae * Bb + Af * Bf + Ag * Bj
        self.g = Ae * Bc + Af * Bg + Ag * Bk
        self.i = Ai * Ba + Aj * Be + Ak * Bi
        self.j = Ai * Bb + Aj * Bf + Ak * Bj
        self.k = Ai * Bc + Aj * Bg + Ak * Bk
        return self

    def identity(self):
        self.a = self.f = self.k = 1.
        self.b = self.c = self.e = self.g = self.i = self.j = 0
        return self

    def scale(self, x, y):
        self *= Matrix3.new_scale(x, y)
        return self

    def translate(self, x, y):
        self *= Matrix3.new_translate(x, y)
        return self 

    def rotate(self, angle):
        self *= Matrix3.new_rotate(angle)
        return self

    # Static constructors
    def new_identity(cls):
        self = cls()
        return self
    new_identity = classmethod(new_identity)

    def new_scale(cls, x, y):
        self = cls()
        self.a = x
        self.f = y
        return self
    new_scale = classmethod(new_scale)

    def new_translate(cls, x, y):
        self = cls()
        self.c = x
        self.g = y
        return self
    new_translate = classmethod(new_translate)

    def new_rotate(cls, angle):
        self = cls()
        s = math.sin(angle)
        c = math.cos(angle)
        self.a = self.f = c
        self.b = -s
        self.e = s
        return self
    new_rotate = classmethod(new_rotate)

    def determinant(self):
        return (self.a*self.f*self.k 
                + self.b*self.g*self.i 
                + self.c*self.e*self.j 
                - self.a*self.g*self.j 
                - self.b*self.e*self.k 
                - self.c*self.f*self.i)

    def inverse(self):
        tmp = Matrix3()
        d = self.determinant()

        if abs(d) < 0.001:
            # No inverse, return identity
            return tmp
        else:
            d = 1.0 / d

            tmp.a = d * (self.f*self.k - self.g*self.j)
            tmp.b = d * (self.c*self.j - self.b*self.k)
            tmp.c = d * (self.b*self.g - self.c*self.f)
            tmp.e = d * (self.g*self.i - self.e*self.k)
            tmp.f = d * (self.a*self.k - self.c*self.i)
            tmp.g = d * (self.c*self.e - self.a*self.g)
            tmp.i = d * (self.e*self.j - self.f*self.i)
            tmp.j = d * (self.b*self.i - self.a*self.j)
            tmp.k = d * (self.a*self.f - self.b*self.e)

            return tmp

# a b c d
# e f g h
# i j k l
# m n o p

class Matrix4:
    __slots__ = list('abcdefghijklmnop')

    def __init__(self):
        self.identity()

    def __copy__(self):
        M = Matrix4()
        M.a = self.a
        M.b = self.b
        M.c = self.c
        M.d = self.d
        M.e = self.e 
        M.f = self.f
        M.g = self.g
        M.h = self.h
        M.i = self.i
        M.j = self.j
        M.k = self.k
        M.l = self.l
        M.m = self.m
        M.n = self.n
        M.o = self.o
        M.p = self.p
        return M

    copy = __copy__


    def __repr__(self):
        return ('Matrix4([% 8.2f % 8.2f % 8.2f % 8.2f\n'  \
                '         % 8.2f % 8.2f % 8.2f % 8.2f\n'  \
                '         % 8.2f % 8.2f % 8.2f % 8.2f\n'  \
                '         % 8.2f % 8.2f % 8.2f % 8.2f])') \
                % (self.a, self.b, self.c, self.d,
                   self.e, self.f, self.g, self.h,
                   self.i, self.j, self.k, self.l,
                   self.m, self.n, self.o, self.p)

    def __getitem__(self, key):
        return [self.a, self.e, self.i, self.m,
                self.b, self.f, self.j, self.n,
                self.c, self.g, self.k, self.o,
                self.d, self.h, self.l, self.p][key]

    def __setitem__(self, key, value):
        L = self[:]
        L[key] = value
        (self.a, self.e, self.i, self.m,
         self.b, self.f, self.j, self.n,
         self.c, self.g, self.k, self.o,
         self.d, self.h, self.l, self.p) = L

    def __mul__(self, other):
        if isinstance(other, Matrix4):
            # Cache attributes in local vars (see Matrix3.__mul__).
            Aa = self.a
            Ab = self.b
            Ac = self.c
            Ad = self.d
            Ae = self.e
            Af = self.f
            Ag = self.g
            Ah = self.h
            Ai = self.i
            Aj = self.j
            Ak = self.k
            Al = self.l
            Am = self.m
            An = self.n
            Ao = self.o
            Ap = self.p
            Ba = other.a
            Bb = other.b
            Bc = other.c
            Bd = other.d
            Be = other.e
            Bf = other.f
            Bg = other.g
            Bh = other.h
            Bi = other.i
            Bj = other.j
            Bk = other.k
            Bl = other.l
            Bm = other.m
            Bn = other.n
            Bo = other.o
            Bp = other.p
            C = Matrix4()
            C.a = Aa * Ba + Ab * Be + Ac * Bi + Ad * Bm
            C.b = Aa * Bb + Ab * Bf + Ac * Bj + Ad * Bn
            C.c = Aa * Bc + Ab * Bg + Ac * Bk + Ad * Bo
            C.d = Aa * Bd + Ab * Bh + Ac * Bl + Ad * Bp
            C.e = Ae * Ba + Af * Be + Ag * Bi + Ah * Bm
            C.f = Ae * Bb + Af * Bf + Ag * Bj + Ah * Bn
            C.g = Ae * Bc + Af * Bg + Ag * Bk + Ah * Bo
            C.h = Ae * Bd + Af * Bh + Ag * Bl + Ah * Bp
            C.i = Ai * Ba + Aj * Be + Ak * Bi + Al * Bm
            C.j = Ai * Bb + Aj * Bf + Ak * Bj + Al * Bn
            C.k = Ai * Bc + Aj * Bg + Ak * Bk + Al * Bo
            C.l = Ai * Bd + Aj * Bh + Ak * Bl + Al * Bp
            C.m = Am * Ba + An * Be + Ao * Bi + Ap * Bm
            C.n = Am * Bb + An * Bf + Ao * Bj + Ap * Bn
            C.o = Am * Bc + An * Bg + Ao * Bk + Ap * Bo
            C.p = Am * Bd + An * Bh + Ao * Bl + Ap * Bp
            return C
        elif isinstance(other, Point3):
            A = self
            B = other
            P = Point3(0, 0, 0)
            P.x = A.a * B.x + A.b * B.y + A.c * B.z + A.d
            P.y = A.e * B.x + A.f * B.y + A.g * B.z + A.h
            P.z = A.i * B.x + A.j * B.y + A.k * B.z + A.l
            return P
        elif isinstance(other, Vector3):
            A = self
            B = other
            V = Vector3(0, 0, 0)
            V.x = A.a * B.x + A.b * B.y + A.c * B.z
            V.y = A.e * B.x + A.f * B.y + A.g * B.z
            V.z = A.i * B.x + A.j * B.y + A.k * B.z
            return V
        else:
            other = other.copy()
            other._apply_transform(self)
            return other

    def __imul__(self, other):
        assert isinstance(other, Matrix4)
        # Cache attributes in local vars (see Matrix3.__mul__).
        Aa = self.a
        Ab = self.b
        Ac = self.c
        Ad = self.d
        Ae = self.e
        Af = self.f
        Ag = self.g
        Ah = self.h
        Ai = self.i
        Aj = self.j
        Ak = self.k
        Al = self.l
        Am = self.m
        An = self.n
        Ao = self.o
        Ap = self.p
        Ba = other.a
        Bb = other.b
        Bc = other.c
        Bd = other.d
        Be = other.e
        Bf = other.f
        Bg = other.g
        Bh = other.h
        Bi = other.i
        Bj = other.j
        Bk = other.k
        Bl = other.l
        Bm = other.m
        Bn = other.n
        Bo = other.o
        Bp = other.p
        self.a = Aa * Ba + Ab * Be + Ac * Bi + Ad * Bm
        self.b = Aa * Bb + Ab * Bf + Ac * Bj + Ad * Bn
        self.c = Aa * Bc + Ab * Bg + Ac * Bk + Ad * Bo
        self.d = Aa * Bd + Ab * Bh + Ac * Bl + Ad * Bp
        self.e = Ae * Ba + Af * Be + Ag * Bi + Ah * Bm
        self.f = Ae * Bb + Af * Bf + Ag * Bj + Ah * Bn
        self.g = Ae * Bc + Af * Bg + Ag * Bk + Ah * Bo
        self.h = Ae * Bd + Af * Bh + Ag * Bl + Ah * Bp
        self.i = Ai * Ba + Aj * Be + Ak * Bi + Al * Bm
        self.j = Ai * Bb + Aj * Bf + Ak * Bj + Al * Bn
        self.k = Ai * Bc + Aj * Bg + Ak * Bk + Al * Bo
        self.l = Ai * Bd + Aj * Bh + Ak * Bl + Al * Bp
        self.m = Am * Ba + An * Be + Ao * Bi + Ap * Bm
        self.n = Am * Bb + An * Bf + Ao * Bj + Ap * Bn
        self.o = Am * Bc + An * Bg + Ao * Bk + Ap * Bo
        self.p = Am * Bd + An * Bh + Ao * Bl + Ap * Bp
        return self

    def transform(self, other):
        A = self
        B = other
        P = Point3(0, 0, 0)
        P.x = A.a * B.x + A.b * B.y + A.c * B.z + A.d
        P.y = A.e * B.x + A.f * B.y + A.g * B.z + A.h
        P.z = A.i * B.x + A.j * B.y + A.k * B.z + A.l
        w =   A.m * B.x + A.n * B.y + A.o * B.z + A.p
        if w != 0:
            P.x /= w
            P.y /= w
            P.z /= w
        return P

    def identity(self):
        self.a = self.f = self.k = self.p = 1.
        self.b = self.c = self.d = self.e = self.g = self.h = \
        self.i = self.j = self.l = self.m = self.n = self.o = 0
        return self

    def scale(self, x, y, z):
        self *= Matrix4.new_scale(x, y, z)
        return self

    def translate(self, x, y, z):
        self *= Matrix4.new_translate(x, y, z)
        return self 

    def rotatex(self, angle):
        self *= Matrix4.new_rotatex(angle)
        return self

    def rotatey(self, angle):
        self *= Matrix4.new_rotatey(angle)
        return self

    def rotatez(self, angle):
        self *= Matrix4.new_rotatez(angle)
        return self

    def rotate_axis(self, angle, axis):
        self *= Matrix4.new_rotate_axis(angle, axis)
        return self

    def rotate_euler(self, heading, attitude, bank):
        self *= Matrix4.new_rotate_euler(heading, attitude, bank)
        return self

    def rotate_triple_axis(self, x, y, z):
        self *= Matrix4.new_rotate_triple_axis(x, y, z)
        return self

    def transpose(self):
        (self.a, self.e, self.i, self.m,
         self.b, self.f, self.j, self.n,
         self.c, self.g, self.k, self.o,
         self.d, self.h, self.l, self.p) = \
        (self.a, self.b, self.c, self.d,
         self.e, self.f, self.g, self.h,
         self.i, self.j, self.k, self.l,
         self.m, self.n, self.o, self.p)

    def transposed(self):
        M = self.copy()
        M.transpose()
        return M

    # Static constructors
    def new(cls, *values):
        M = cls()
        M[:] = values
        return M
    new = classmethod(new)

    def new_identity(cls):
        self = cls()
        return self
    new_identity = classmethod(new_identity)

    def new_scale(cls, x, y, z):
        self = cls()
        self.a = x
        self.f = y
        self.k = z
        return self
    new_scale = classmethod(new_scale)

    def new_translate(cls, x, y, z):
        self = cls()
        self.d = x
        self.h = y
        self.l = z
        return self
    new_translate = classmethod(new_translate)

    def new_rotatex(cls, angle):
        self = cls()
        s = math.sin(angle)
        c = math.cos(angle)
        self.f = self.k = c
        self.g = -s
        self.j = s
        return self
    new_rotatex = classmethod(new_rotatex)

    def new_rotatey(cls, angle):
        self = cls()
        s = math.sin(angle)
        c = math.cos(angle)
        self.a = self.k = c
        self.c = s
        self.i = -s
        return self    
    new_rotatey = classmethod(new_rotatey)
    
    def new_rotatez(cls, angle):
        self = cls()
        s = math.sin(angle)
        c = math.cos(angle)
        self.a = self.f = c
        self.b = -s
        self.e = s
        return self
    new_rotatez = classmethod(new_rotatez)

    def new_rotate_axis(cls, angle, axis):
        assert(isinstance(axis, Vector3))
        vector = axis.normalized()
        x = vector.x
        y = vector.y
        z = vector.z

        self = cls()
        s = math.sin(angle)
        c = math.cos(angle)
        c1 = 1. - c
        
        # from the glRotate man page
        self.a = x * x * c1 + c
        self.b = x * y * c1 - z * s
        self.c = x * z * c1 + y * s
        self.e = y * x * c1 + z * s
        self.f = y * y * c1 + c
        self.g = y * z * c1 - x * s
        self.i = x * z * c1 - y * s
        self.j = y * z * c1 + x * s
        self.k = z * z * c1 + c
        return self
    new_rotate_axis = classmethod(new_rotate_axis)

    def new_rotate_euler(cls, heading, attitude, bank):
        # from http://www.euclideanspace.com/
        ch = math.cos(heading)
        sh = math.sin(heading)
        ca = math.cos(attitude)
        sa = math.sin(attitude)
        cb = math.cos(bank)
        sb = math.sin(bank)

        self = cls()
        self.a = ch * ca
        self.b = sh * sb - ch * sa * cb
        self.c = ch * sa * sb + sh * cb
        self.e = sa
        self.f = ca * cb
        self.g = -ca * sb
        self.i = -sh * ca
        self.j = sh * sa * cb + ch * sb
        self.k = -sh * sa * sb + ch * cb
        return self
    new_rotate_euler = classmethod(new_rotate_euler)

    def new_rotate_triple_axis(cls, x, y, z):
      m = cls()
      
      m.a, m.b, m.c = x.x, y.x, z.x
      m.e, m.f, m.g = x.y, y.y, z.y
      m.i, m.j, m.k = x.z, y.z, z.z
      
      return m
    new_rotate_triple_axis = classmethod(new_rotate_triple_axis)

    def new_look_at(cls, eye, at, up):
      z = (eye - at).normalized()
      x = up.cross(z).normalized()
      y = z.cross(x)
      
      m = cls.new_rotate_triple_axis(x, y, z)
      m.d, m.h, m.l = eye.x, eye.y, eye.z
      return m
    new_look_at = classmethod(new_look_at)
    
    def new_perspective(cls, fov_y, aspect, near, far):
        # from the gluPerspective man page
        f = 1 / math.tan(fov_y / 2)
        self = cls()
        assert near != 0.0 and near != far
        self.a = f / aspect
        self.f = f
        self.k = (far + near) / (near - far)
        self.l = 2 * far * near / (near - far)
        self.o = -1
        self.p = 0
        return self
    new_perspective = classmethod(new_perspective)

    def determinant(self):
        return ((self.a * self.f - self.e * self.b)
              * (self.k * self.p - self.o * self.l)
              - (self.a * self.j - self.i * self.b)
              * (self.g * self.p - self.o * self.h)
              + (self.a * self.n - self.m * self.b)
              * (self.g * self.l - self.k * self.h)
              + (self.e * self.j - self.i * self.f)
              * (self.c * self.p - self.o * self.d)
              - (self.e * self.n - self.m * self.f)
              * (self.c * self.l - self.k * self.d)
              + (self.i * self.n - self.m * self.j)
              * (self.c * self.h - self.g * self.d))

    def inverse(self):
        tmp = Matrix4()
        d = self.determinant();

        if abs(d) < 0.001:
            # No inverse, return identity
            return tmp
        else:
            d = 1.0 / d;

            tmp.a = d * (self.f * (self.k * self.p - self.o * self.l) + self.j * (self.o * self.h - self.g * self.p) + self.n * (self.g * self.l - self.k * self.h));
            tmp.e = d * (self.g * (self.i * self.p - self.m * self.l) + self.k * (self.m * self.h - self.e * self.p) + self.o * (self.e * self.l - self.i * self.h));
            tmp.i = d * (self.h * (self.i * self.n - self.m * self.j) + self.l * (self.m * self.f - self.e * self.n) + self.p * (self.e * self.j - self.i * self.f));
            tmp.m = d * (self.e * (self.n * self.k - self.j * self.o) + self.i * (self.f * self.o - self.n * self.g) + self.m * (self.j * self.g - self.f * self.k));
            
            tmp.b = d * (self.j * (self.c * self.p - self.o * self.d) + self.n * (self.k * self.d - self.c * self.l) + self.b * (self.o * self.l - self.k * self.p));
            tmp.f = d * (self.k * (self.a * self.p - self.m * self.d) + self.o * (self.i * self.d - self.a * self.l) + self.c * (self.m * self.l - self.i * self.p));
            tmp.j = d * (self.l * (self.a * self.n - self.m * self.b) + self.p * (self.i * self.b - self.a * self.j) + self.d * (self.m * self.j - self.i * self.n));
            tmp.n = d * (self.i * (self.n * self.c - self.b * self.o) + self.m * (self.b * self.k - self.j * self.c) + self.a * (self.j * self.o - self.n * self.k));
            
            tmp.c = d * (self.n * (self.c * self.h - self.g * self.d) + self.b * (self.g * self.p - self.o * self.h) + self.f * (self.o * self.d - self.c * self.p));
            tmp.g = d * (self.o * (self.a * self.h - self.e * self.d) + self.c * (self.e * self.p - self.m * self.h) + self.g * (self.m * self.d - self.a * self.p));
            tmp.k = d * (self.p * (self.a * self.f - self.e * self.b) + self.d * (self.e * self.n - self.m * self.f) + self.h * (self.m * self.b - self.a * self.n));
            tmp.o = d * (self.m * (self.f * self.c - self.b * self.g) + self.a * (self.n * self.g - self.f * self.o) + self.e * (self.b * self.o - self.n * self.c));
            
            tmp.d = d * (self.b * (self.k * self.h - self.g * self.l) + self.f * (self.c * self.l - self.k * self.d) + self.j * (self.g * self.d - self.c * self.h));
            tmp.h = d * (self.c * (self.i * self.h - self.e * self.l) + self.g * (self.a * self.l - self.i * self.d) + self.k * (self.e * self.d - self.a * self.h));
            tmp.l = d * (self.d * (self.i * self.f - self.e * self.j) + self.h * (self.a * self.j - self.i * self.b) + self.l * (self.e * self.b - self.a * self.f));
            tmp.p = d * (self.a * (self.f * self.k - self.j * self.g) + self.e * (self.j * self.c - self.b * self.k) + self.i * (self.b * self.g - self.f * self.c));

        return tmp;
        

class Quaternion:
    # All methods and naming conventions based off 
    # http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions

    # w is the real part, (x, y, z) are the imaginary parts
    __slots__ = ['w', 'x', 'y', 'z']

    def __init__(self, w=1, x=0, y=0, z=0):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __copy__(self):
        Q = Quaternion()
        Q.w = self.w
        Q.x = self.x
        Q.y = self.y
        Q.z = self.z
        return Q

    copy = __copy__

    def __repr__(self):
        return 'Quaternion(real=%.2f, imag=<%.2f, %.2f, %.2f>)' % \
            (self.w, self.x, self.y, self.z)

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            Ax = self.x
            Ay = self.y
            Az = self.z
            Aw = self.w
            Bx = other.x
            By = other.y
            Bz = other.z
            Bw = other.w
            Q = Quaternion()
            Q.x =  Ax * Bw + Ay * Bz - Az * By + Aw * Bx    
            Q.y = -Ax * Bz + Ay * Bw + Az * Bx + Aw * By
            Q.z =  Ax * By - Ay * Bx + Az * Bw + Aw * Bz
            Q.w = -Ax * Bx - Ay * By - Az * Bz + Aw * Bw
            return Q
        elif isinstance(other, Vector3):
            w = self.w
            x = self.x
            y = self.y
            z = self.z
            Vx = other.x
            Vy = other.y
            Vz = other.z
            ww = w * w
            w2 = w * 2
            wx2 = w2 * x
            wy2 = w2 * y
            wz2 = w2 * z
            xx = x * x
            x2 = x * 2
            xy2 = x2 * y
            xz2 = x2 * z
            yy = y * y
            yz2 = 2 * y * z
            zz = z * z
            return other.__class__(\
               ww * Vx + wy2 * Vz - wz2 * Vy + \
               xx * Vx + xy2 * Vy + xz2 * Vz - \
               zz * Vx - yy * Vx,
               xy2 * Vx + yy * Vy + yz2 * Vz + \
               wz2 * Vx - zz * Vy + ww * Vy - \
               wx2 * Vz - xx * Vy,
               xz2 * Vx + yz2 * Vy + \
               zz * Vz - wy2 * Vx - yy * Vz + \
               wx2 * Vy - xx * Vz + ww * Vz)
        else:
            other = other.copy()
            other._apply_transform(self)
            return other

    def __imul__(self, other):
        assert isinstance(other, Quaternion)
        Ax = self.x
        Ay = self.y
        Az = self.z
        Aw = self.w
        Bx = other.x
        By = other.y
        Bz = other.z
        Bw = other.w
        self.x =  Ax * Bw + Ay * Bz - Az * By + Aw * Bx    
        self.y = -Ax * Bz + Ay * Bw + Az * Bx + Aw * By
        self.z =  Ax * By - Ay * Bx + Az * Bw + Aw * Bz
        self.w = -Ax * Bx - Ay * By - Az * Bz + Aw * Bw
        return self

    def __abs__(self):
        return math.sqrt(self.w ** 2 + \
                         self.x ** 2 + \
                         self.y ** 2 + \
                         self.z ** 2)

    magnitude = __abs__

    def magnitude_squared(self):
        return self.w ** 2 + \
               self.x ** 2 + \
               self.y ** 2 + \
               self.z ** 2 

    def identity(self):
        self.w = 1
        self.x = 0
        self.y = 0
        self.z = 0
        return self

    def rotate_axis(self, angle, axis):
        self *= Quaternion.new_rotate_axis(angle, axis)
        return self

    def rotate_euler(self, heading, attitude, bank):
        self *= Quaternion.new_rotate_euler(heading, attitude, bank)
        return self

    def rotate_matrix(self, m):
        self *= Quaternion.new_rotate_matrix(m)
        return self

    def conjugated(self):
        Q = Quaternion()
        Q.w = self.w
        Q.x = -self.x
        Q.y = -self.y
        Q.z = -self.z
        return Q

    def normalize(self):
        d = self.magnitude()
        if d != 0:
            self.w /= d
            self.x /= d
            self.y /= d
            self.z /= d
        return self

    def normalized(self):
        d = self.magnitude()
        if d != 0:
            Q = Quaternion()
            Q.w = self.w / d
            Q.x = self.x / d
            Q.y = self.y / d
            Q.z = self.z / d
            return Q
        else:
            return self.copy()

    def get_angle_axis(self):
        if self.w > 1:
            self = self.normalized()
        angle = 2 * math.acos(self.w)
        s = math.sqrt(1 - self.w ** 2)
        if s < 0.001:
            return angle, Vector3(1, 0, 0)
        else:
            return angle, Vector3(self.x / s, self.y / s, self.z / s)

    def get_euler(self):
        t = self.x * self.y + self.z * self.w
        if t > 0.4999:
            heading = 2 * math.atan2(self.x, self.w)
            attitude = math.pi / 2
            bank = 0
        elif t < -0.4999:
            heading = -2 * math.atan2(self.x, self.w)
            attitude = -math.pi / 2
            bank = 0
        else:
            sqx = self.x ** 2
            sqy = self.y ** 2
            sqz = self.z ** 2
            heading = math.atan2(2 * self.y * self.w - 2 * self.x * self.z,
                                 1 - 2 * sqy - 2 * sqz)
            attitude = math.asin(2 * t)
            bank = math.atan2(2 * self.x * self.w - 2 * self.y * self.z,
                              1 - 2 * sqx - 2 * sqz)
        return heading, attitude, bank

    def get_matrix(self):
        xx = self.x ** 2
        xy = self.x * self.y
        xz = self.x * self.z
        xw = self.x * self.w
        yy = self.y ** 2
        yz = self.y * self.z
        yw = self.y * self.w
        zz = self.z ** 2
        zw = self.z * self.w
        M = Matrix4()
        M.a = 1 - 2 * (yy + zz)
        M.b = 2 * (xy - zw)
        M.c = 2 * (xz + yw)
        M.e = 2 * (xy + zw)
        M.f = 1 - 2 * (xx + zz)
        M.g = 2 * (yz - xw)
        M.i = 2 * (xz - yw)
        M.j = 2 * (yz + xw)
        M.k = 1 - 2 * (xx + yy)
        return M

    # Static constructors
    def new_identity(cls):
        return cls()
    new_identity = classmethod(new_identity)

    def new_rotate_axis(cls, angle, axis):
        assert(isinstance(axis, Vector3))
        axis = axis.normalized()
        s = math.sin(angle / 2)
        Q = cls()
        Q.w = math.cos(angle / 2)
        Q.x = axis.x * s
        Q.y = axis.y * s
        Q.z = axis.z * s
        return Q
    new_rotate_axis = classmethod(new_rotate_axis)

    def new_rotate_euler(cls, heading, attitude, bank):
        Q = cls()
        c1 = math.cos(heading / 2)
        s1 = math.sin(heading / 2)
        c2 = math.cos(attitude / 2)
        s2 = math.sin(attitude / 2)
        c3 = math.cos(bank / 2)
        s3 = math.sin(bank / 2)

        Q.w = c1 * c2 * c3 - s1 * s2 * s3
        Q.x = s1 * s2 * c3 + c1 * c2 * s3
        Q.y = s1 * c2 * c3 + c1 * s2 * s3
        Q.z = c1 * s2 * c3 - s1 * c2 * s3
        return Q
    new_rotate_euler = classmethod(new_rotate_euler)
    
    def new_rotate_matrix(cls, m):
      if m[0*4 + 0] + m[1*4 + 1] + m[2*4 + 2] > 0.00000001:
        t = m[0*4 + 0] + m[1*4 + 1] + m[2*4 + 2] + 1.0
        s = 0.5/math.sqrt(t)
        
        return cls(
          s*t,
          (m[1*4 + 2] - m[2*4 + 1])*s,
          (m[2*4 + 0] - m[0*4 + 2])*s,
          (m[0*4 + 1] - m[1*4 + 0])*s
          )
        
      elif m[0*4 + 0] > m[1*4 + 1] and m[0*4 + 0] > m[2*4 + 2]:
        t = m[0*4 + 0] - m[1*4 + 1] - m[2*4 + 2] + 1.0
        s = 0.5/math.sqrt(t)
        
        return cls(
          (m[1*4 + 2] - m[2*4 + 1])*s,
          s*t,
          (m[0*4 + 1] + m[1*4 + 0])*s,
          (m[2*4 + 0] + m[0*4 + 2])*s
          )
        
      elif m[1*4 + 1] > m[2*4 + 2]:
        t = -m[0*4 + 0] + m[1*4 + 1] - m[2*4 + 2] + 1.0
        s = 0.5/math.sqrt(t)
        
        return cls(
          (m[2*4 + 0] - m[0*4 + 2])*s,
          (m[0*4 + 1] + m[1*4 + 0])*s,
          s*t,
          (m[1*4 + 2] + m[2*4 + 1])*s
          )
        
      else:
        t = -m[0*4 + 0] - m[1*4 + 1] + m[2*4 + 2] + 1.0
        s = 0.5/math.sqrt(t)
        
        return cls(
          (m[0*4 + 1] - m[1*4 + 0])*s,
          (m[2*4 + 0] + m[0*4 + 2])*s,
          (m[1*4 + 2] + m[2*4 + 1])*s,
          s*t
          )
    new_rotate_matrix = classmethod(new_rotate_matrix)
    
    def new_interpolate(cls, q1, q2, t):
        assert isinstance(q1, Quaternion) and isinstance(q2, Quaternion)
        Q = cls()

        costheta = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z
        if costheta < 0.:
            costheta = -costheta
            q1 = q1.conjugated()
        elif costheta > 1:
            costheta = 1

        theta = math.acos(costheta)
        if abs(theta) < 0.01:
            Q.w = q2.w
            Q.x = q2.x
            Q.y = q2.y
            Q.z = q2.z
            return Q

        sintheta = math.sqrt(1.0 - costheta * costheta)
        if abs(sintheta) < 0.01:
            Q.w = (q1.w + q2.w) * 0.5
            Q.x = (q1.x + q2.x) * 0.5
            Q.y = (q1.y + q2.y) * 0.5
            Q.z = (q1.z + q2.z) * 0.5
            return Q

        ratio1 = math.sin((1 - t) * theta) / sintheta
        ratio2 = math.sin(t * theta) / sintheta

        Q.w = q1.w * ratio1 + q2.w * ratio2
        Q.x = q1.x * ratio1 + q2.x * ratio2
        Q.y = q1.y * ratio1 + q2.y * ratio2
        Q.z = q1.z * ratio1 + q2.z * ratio2
        return Q
    new_interpolate = classmethod(new_interpolate)

# Geometry
# Much maths thanks to Paul Bourke, http://astronomy.swin.edu.au/~pbourke
# ---------------------------------------------------------------------------

class Geometry:
    def _connect_unimplemented(self, other):
        raise AttributeError( 'Cannot connect %s to %s' % \
            (self.__class__, other.__class__))

    def _intersect_unimplemented(self, other):
        raise AttributeError( 'Cannot intersect %s and %s' % \
            (self.__class__, other.__class__))

    _intersect_point2 = _intersect_unimplemented
    _intersect_line2 = _intersect_unimplemented
    _intersect_circle = _intersect_unimplemented
    _connect_point2 = _connect_unimplemented
    _connect_line2 = _connect_unimplemented
    _connect_circle = _connect_unimplemented

    _intersect_point3 = _intersect_unimplemented
    _intersect_line3 = _intersect_unimplemented
    _intersect_sphere = _intersect_unimplemented
    _intersect_plane = _intersect_unimplemented
    _connect_point3 = _connect_unimplemented
    _connect_line3 = _connect_unimplemented
    _connect_sphere = _connect_unimplemented
    _connect_plane = _connect_unimplemented

    def intersect(self, other):
        raise NotImplementedError

    def connect(self, other):
        raise NotImplementedError

    def distance(self, other):
        c = self.connect(other)
        if c:
            return c.length
        return 0.0

def _intersect_point2_circle(P, C):
    return abs(P - C.c) <= C.r
    
def _intersect_line2_line2(A, B):
    d = B.v.y * A.v.x - B.v.x * A.v.y
    if d == 0:
        return None

    dy = A.p.y - B.p.y
    dx = A.p.x - B.p.x
    ua = (B.v.x * dy - B.v.y * dx) / d
    if not A._u_in(ua):
        return None
    ub = (A.v.x * dy - A.v.y * dx) / d
    if not B._u_in(ub):
        return None

    return Point2(A.p.x + ua * A.v.x,
                  A.p.y + ua * A.v.y)

def _intersect_line2_circle(L, C):
    a = L.v.magnitude_squared()
    b = 2 * (L.v.x * (L.p.x - C.c.x) + \
             L.v.y * (L.p.y - C.c.y))
    c = C.c.magnitude_squared() + \
        L.p.magnitude_squared() - \
        2 * C.c.dot(L.p) - \
        C.r ** 2
    det = b ** 2 - 4 * a * c
    if det < 0:
        return None
    sq = math.sqrt(det)
    u1 = (-b + sq) / (2 * a)
    u2 = (-b - sq) / (2 * a)
    if not L._u_in(u1):
        u1 = max(min(u1, 1.0), 0.0)
    if not L._u_in(u2):
        u2 = max(min(u2, 1.0), 0.0)

    # Tangent
    if u1 == u2:
        return Point2(L.p.x + u1 * L.v.x,
                      L.p.y + u1 * L.v.y)

    return LineSegment2(Point2(L.p.x + u1 * L.v.x,
                               L.p.y + u1 * L.v.y),
                        Point2(L.p.x + u2 * L.v.x,
                               L.p.y + u2 * L.v.y))

def _connect_point2_line2(P, L):
    d = L.v.magnitude_squared()
    assert d != 0
    u = ((P.x - L.p.x) * L.v.x + \
         (P.y - L.p.y) * L.v.y) / d
    if not L._u_in(u):
        u = max(min(u, 1.0), 0.0)
    return LineSegment2(P, 
                        Point2(L.p.x + u * L.v.x,
                               L.p.y + u * L.v.y))

def _connect_point2_circle(P, C):
    v = P - C.c
    v.normalize()
    v *= C.r
    return LineSegment2(P, Point2(C.c.x + v.x, C.c.y + v.y))

def _connect_line2_line2(A, B):
    d = B.v.y * A.v.x - B.v.x * A.v.y
    if d == 0:
        # Parallel, connect an endpoint with a line
        if isinstance(B, Ray2) or isinstance(B, LineSegment2):
            p1, p2 = _connect_point2_line2(B.p, A)
            return p2, p1
        # No endpoint (or endpoint is on A), possibly choose arbitrary point
        # on line.
        return _connect_point2_line2(A.p, B)

    dy = A.p.y - B.p.y
    dx = A.p.x - B.p.x
    ua = (B.v.x * dy - B.v.y * dx) / d
    if not A._u_in(ua):
        ua = max(min(ua, 1.0), 0.0)
    ub = (A.v.x * dy - A.v.y * dx) / d
    if not B._u_in(ub):
        ub = max(min(ub, 1.0), 0.0)

    return LineSegment2(Point2(A.p.x + ua * A.v.x, A.p.y + ua * A.v.y),
                        Point2(B.p.x + ub * B.v.x, B.p.y + ub * B.v.y))

def _connect_circle_line2(C, L):
    d = L.v.magnitude_squared()
    assert d != 0
    u = ((C.c.x - L.p.x) * L.v.x + (C.c.y - L.p.y) * L.v.y) / d
    if not L._u_in(u):
        u = max(min(u, 1.0), 0.0)
    point = Point2(L.p.x + u * L.v.x, L.p.y + u * L.v.y)
    v = (point - C.c)
    v.normalize()
    v *= C.r
    return LineSegment2(Point2(C.c.x + v.x, C.c.y + v.y), point)

def _connect_circle_circle(A, B):
    v = B.c - A.c
    d = v.magnitude()
    if A.r >= B.r and d < A.r:
        #centre B inside A
        s1,s2 = +1, +1
    elif B.r > A.r and d < B.r:
        #centre A inside B
        s1,s2 = -1, -1
    elif d >= A.r and d >= B.r:
        s1,s2 = +1, -1
    v.normalize()
    return LineSegment2(Point2(A.c.x + s1 * v.x * A.r, A.c.y + s1 * v.y * A.r),
                        Point2(B.c.x + s2 * v.x * B.r, B.c.y + s2 * v.y * B.r))


class Point2(Vector2, Geometry):
    def __repr__(self):
        return 'Point2(%.2f, %.2f)' % (self.x, self.y)

    def intersect(self, other):
        return other._intersect_point2(self)

    def _intersect_circle(self, other):
        return _intersect_point2_circle(self, other)

    def connect(self, other):
        return other._connect_point2(self)

    def _connect_point2(self, other):
        return LineSegment2(other, self)
    
    def _connect_line2(self, other):
        c = _connect_point2_line2(self, other)
        if c:
            return c._swap()

    def _connect_circle(self, other):
        c = _connect_point2_circle(self, other)
        if c:
            return c._swap()

class Line2(Geometry):
    __slots__ = ['p', 'v']

    def __init__(self, *args):
        if len(args) == 3:
            assert isinstance(args[0], Point2) and \
                   isinstance(args[1], Vector2) and \
                   type(args[2]) == float
            self.p = args[0].copy()
            self.v = args[1] * args[2] / abs(args[1])
        elif len(args) == 2:
            if isinstance(args[0], Point2) and isinstance(args[1], Point2):
                self.p = args[0].copy()
                self.v = args[1] - args[0]
            elif isinstance(args[0], Point2) and isinstance(args[1], Vector2):
                self.p = args[0].copy()
                self.v = args[1].copy()
            else:
                raise AttributeError( '%r' % (args,))
        elif len(args) == 1:
            if isinstance(args[0], Line2):
                self.p = args[0].p.copy()
                self.v = args[0].v.copy()
            else:
                raise AttributeError( '%r' % (args,))
        else:
            raise AttributeError( '%r' % (args,))
        
        if not self.v:
            raise AttributeError( 'Line has zero-length vector')

    def __copy__(self):
        return self.__class__(self.p, self.v)

    copy = __copy__

    def __repr__(self):
        return 'Line2(<%.2f, %.2f> + u<%.2f, %.2f>)' % \
            (self.p.x, self.p.y, self.v.x, self.v.y)

    p1 = property(lambda self: self.p)
    p2 = property(lambda self: Point2(self.p.x + self.v.x, 
                                      self.p.y + self.v.y))

    def _apply_transform(self, t):
        self.p = t * self.p
        self.v = t * self.v

    def _u_in(self, u):
        return True

    def intersect(self, other):
        return other._intersect_line2(self)

    def _intersect_line2(self, other):
        return _intersect_line2_line2(self, other)

    def _intersect_circle(self, other):
        return _intersect_line2_circle(self, other)

    def connect(self, other):
        return other._connect_line2(self)

    def _connect_point2(self, other):
        return _connect_point2_line2(other, self)

    def _connect_line2(self, other):
        return _connect_line2_line2(other, self)

    def _connect_circle(self, other):
        return _connect_circle_line2(other, self)

class Ray2(Line2):
    def __repr__(self):
        return 'Ray2(<%.2f, %.2f> + u<%.2f, %.2f>)' % \
            (self.p.x, self.p.y, self.v.x, self.v.y)

    def _u_in(self, u):
        return u >= 0.0

class LineSegment2(Line2):
    def __repr__(self):
        return 'LineSegment2(<%.2f, %.2f> to <%.2f, %.2f>)' % \
            (self.p.x, self.p.y, self.p.x + self.v.x, self.p.y + self.v.y)

    def _u_in(self, u):
        return u >= 0.0 and u <= 1.0

    def __abs__(self):
        return abs(self.v)

    def magnitude_squared(self):
        return self.v.magnitude_squared()

    def _swap(self):
        # used by connect methods to switch order of points
        self.p = self.p2
        self.v *= -1
        return self

    length = property(lambda self: abs(self.v))

class Circle(Geometry):
    __slots__ = ['c', 'r']

    def __init__(self, center, radius):
        assert isinstance(center, Vector2) and type(radius) == float
        self.c = center.copy()
        self.r = radius

    def __copy__(self):
        return self.__class__(self.c, self.r)

    copy = __copy__

    def __repr__(self):
        return 'Circle(<%.2f, %.2f>, radius=%.2f)' % \
            (self.c.x, self.c.y, self.r)

    def _apply_transform(self, t):
        self.c = t * self.c

    def intersect(self, other):
        return other._intersect_circle(self)

    def _intersect_point2(self, other):
        return _intersect_point2_circle(other, self)

    def _intersect_line2(self, other):
        return _intersect_line2_circle(other, self)

    def connect(self, other):
        return other._connect_circle(self)

    def _connect_point2(self, other):
        return _connect_point2_circle(other, self)

    def _connect_line2(self, other):
        c = _connect_circle_line2(self, other)
        if c:
            return c._swap()

    def _connect_circle(self, other):
        return _connect_circle_circle(other, self)

# 3D Geometry
# -------------------------------------------------------------------------

def _connect_point3_line3(P, L):
    d = L.v.magnitude_squared()
    assert d != 0
    u = ((P.x - L.p.x) * L.v.x + \
         (P.y - L.p.y) * L.v.y + \
         (P.z - L.p.z) * L.v.z) / d
    if not L._u_in(u):
        u = max(min(u, 1.0), 0.0)
    return LineSegment3(P, Point3(L.p.x + u * L.v.x,
                                  L.p.y + u * L.v.y,
                                  L.p.z + u * L.v.z))

def _connect_point3_sphere(P, S):
    v = P - S.c
    v.normalize()
    v *= S.r
    return LineSegment3(P, Point3(S.c.x + v.x, S.c.y + v.y, S.c.z + v.z))

def _connect_point3_plane(p, plane):
    n = plane.n.normalized()
    d = p.dot(plane.n) - plane.k
    return LineSegment3(p, Point3(p.x - n.x * d, p.y - n.y * d, p.z - n.z * d))

def _connect_line3_line3(A, B):
    assert A.v and B.v
    p13 = A.p - B.p
    d1343 = p13.dot(B.v)
    d4321 = B.v.dot(A.v)
    d1321 = p13.dot(A.v)
    d4343 = B.v.magnitude_squared()
    denom = A.v.magnitude_squared() * d4343 - d4321 ** 2
    if denom == 0:
        # Parallel, connect an endpoint with a line
        if isinstance(B, Ray3) or isinstance(B, LineSegment3):
            return _connect_point3_line3(B.p, A)._swap()
        # No endpoint (or endpoint is on A), possibly choose arbitrary
        # point on line.
        return _connect_point3_line3(A.p, B)

    ua = (d1343 * d4321 - d1321 * d4343) / denom
    if not A._u_in(ua):
        ua = max(min(ua, 1.0), 0.0)
    ub = (d1343 + d4321 * ua) / d4343
    if not B._u_in(ub):
        ub = max(min(ub, 1.0), 0.0)
    return LineSegment3(Point3(A.p.x + ua * A.v.x,
                               A.p.y + ua * A.v.y,
                               A.p.z + ua * A.v.z),
                        Point3(B.p.x + ub * B.v.x,
                               B.p.y + ub * B.v.y,
                               B.p.z + ub * B.v.z))

def _connect_line3_plane(L, P):
    d = P.n.dot(L.v)
    if not d:
        # Parallel, choose an endpoint
        return _connect_point3_plane(L.p, P)
    u = (P.k - P.n.dot(L.p)) / d
    if not L._u_in(u):
        # intersects out of range, choose nearest endpoint
        u = max(min(u, 1.0), 0.0)
        return _connect_point3_plane(Point3(L.p.x + u * L.v.x,
                                            L.p.y + u * L.v.y,
                                            L.p.z + u * L.v.z), P)
    # Intersection
    return None

def _connect_sphere_line3(S, L):
    d = L.v.magnitude_squared()
    assert d != 0
    u = ((S.c.x - L.p.x) * L.v.x + \
         (S.c.y - L.p.y) * L.v.y + \
         (S.c.z - L.p.z) * L.v.z) / d
    if not L._u_in(u):
        u = max(min(u, 1.0), 0.0)
    point = Point3(L.p.x + u * L.v.x, L.p.y + u * L.v.y, L.p.z + u * L.v.z)
    v = (point - S.c)
    v.normalize()
    v *= S.r
    return LineSegment3(Point3(S.c.x + v.x, S.c.y + v.y, S.c.z + v.z), 
                        point)

def _connect_sphere_sphere(A, B):
    v = B.c - A.c
    d = v.magnitude()
    if A.r >= B.r and d < A.r:
        #centre B inside A
        s1,s2 = +1, +1
    elif B.r > A.r and d < B.r:
        #centre A inside B
        s1,s2 = -1, -1
    elif d >= A.r and d >= B.r:
        s1,s2 = +1, -1

    v.normalize()
    return LineSegment3(Point3(A.c.x + s1* v.x * A.r,
                               A.c.y + s1* v.y * A.r,
                               A.c.z + s1* v.z * A.r),
                        Point3(B.c.x + s2* v.x * B.r,
                               B.c.y + s2* v.y * B.r,
                               B.c.z + s2* v.z * B.r))

def _connect_sphere_plane(S, P):
    c = _connect_point3_plane(S.c, P)
    if not c:
        return None
    p2 = c.p2
    v = p2 - S.c
    v.normalize()
    v *= S.r
    return LineSegment3(Point3(S.c.x + v.x, S.c.y + v.y, S.c.z + v.z), 
                        p2)

def _connect_plane_plane(A, B):
    if A.n.cross(B.n):
        # Planes intersect
        return None
    else:
        # Planes are parallel, connect to arbitrary point
        return _connect_point3_plane(A._get_point(), B)

def _intersect_point3_sphere(P, S):
    return abs(P - S.c) <= S.r
    
def _intersect_line3_sphere(L, S):
    a = L.v.magnitude_squared()
    b = 2 * (L.v.x * (L.p.x - S.c.x) + \
             L.v.y * (L.p.y - S.c.y) + \
             L.v.z * (L.p.z - S.c.z))
    c = S.c.magnitude_squared() + \
        L.p.magnitude_squared() - \
        2 * S.c.dot(L.p) - \
        S.r ** 2
    det = b ** 2 - 4 * a * c
    if det < 0:
        return None
    sq = math.sqrt(det)
    u1 = (-b + sq) / (2 * a)
    u2 = (-b - sq) / (2 * a)
    if not L._u_in(u1):
        u1 = max(min(u1, 1.0), 0.0)
    if not L._u_in(u2):
        u2 = max(min(u2, 1.0), 0.0)
    return LineSegment3(Point3(L.p.x + u1 * L.v.x,
                               L.p.y + u1 * L.v.y,
                               L.p.z + u1 * L.v.z),
                        Point3(L.p.x + u2 * L.v.x,
                               L.p.y + u2 * L.v.y,
                               L.p.z + u2 * L.v.z))

def _intersect_line3_plane(L, P):
    d = P.n.dot(L.v)
    if not d:
        # Parallel
        return None
    u = (P.k - P.n.dot(L.p)) / d
    if not L._u_in(u):
        return None
    return Point3(L.p.x + u * L.v.x,
                  L.p.y + u * L.v.y,
                  L.p.z + u * L.v.z)

def _intersect_plane_plane(A, B):
    n1_m = A.n.magnitude_squared()
    n2_m = B.n.magnitude_squared()
    n1d2 = A.n.dot(B.n)
    det = n1_m * n2_m - n1d2 ** 2
    if det == 0:
        # Parallel
        return None
    c1 = (A.k * n2_m - B.k * n1d2) / det
    c2 = (B.k * n1_m - A.k * n1d2) / det
    return Line3(Point3(c1 * A.n.x + c2 * B.n.x,
                        c1 * A.n.y + c2 * B.n.y,
                        c1 * A.n.z + c2 * B.n.z), 
                 A.n.cross(B.n))

class Point3(Vector3, Geometry):
    def __repr__(self):
        return 'Point3(%.2f, %.2f, %.2f)' % (self.x, self.y, self.z)

    def intersect(self, other):
        return other._intersect_point3(self)

    def _intersect_sphere(self, other):
        return _intersect_point3_sphere(self, other)

    def connect(self, other):
        return other._connect_point3(self)

    def _connect_point3(self, other):
        if self != other:
            return LineSegment3(other, self)
        return None

    def _connect_line3(self, other):
        c = _connect_point3_line3(self, other)
        if c:
            return c._swap()
        
    def _connect_sphere(self, other):
        c = _connect_point3_sphere(self, other)
        if c:
            return c._swap()

    def _connect_plane(self, other):
        c = _connect_point3_plane(self, other)
        if c:
            return c._swap()

class Line3:
    __slots__ = ['p', 'v']

    def __init__(self, *args):
        if len(args) == 3:
            assert isinstance(args[0], Point3) and \
                   isinstance(args[1], Vector3) and \
                   type(args[2]) == float
            self.p = args[0].copy()
            self.v = args[1] * args[2] / abs(args[1])
        elif len(args) == 2:
            if isinstance(args[0], Point3) and isinstance(args[1], Point3):
                self.p = args[0].copy()
                self.v = args[1] - args[0]
            elif isinstance(args[0], Point3) and isinstance(args[1], Vector3):
                self.p = args[0].copy()
                self.v = args[1].copy()
            else:
                raise AttributeError( '%r' % (args,))
        elif len(args) == 1:
            if isinstance(args[0], Line3):
                self.p = args[0].p.copy()
                self.v = args[0].v.copy()
            else:
                raise AttributeError( '%r' % (args,))
        else:
            raise AttributeError( '%r' % (args,))
        
        # XXX This is annoying.
        #if not self.v:
        #    raise AttributeError, 'Line has zero-length vector'

    def __copy__(self):
        return self.__class__(self.p, self.v)

    copy = __copy__

    def __repr__(self):
        return 'Line3(<%.2f, %.2f, %.2f> + u<%.2f, %.2f, %.2f>)' % \
            (self.p.x, self.p.y, self.p.z, self.v.x, self.v.y, self.v.z)

    p1 = property(lambda self: self.p)
    p2 = property(lambda self: Point3(self.p.x + self.v.x, 
                                      self.p.y + self.v.y,
                                      self.p.z + self.v.z))

    def _apply_transform(self, t):
        self.p = t * self.p
        self.v = t * self.v

    def _u_in(self, u):
        return True

    def intersect(self, other):
        return other._intersect_line3(self)

    def _intersect_sphere(self, other):
        return _intersect_line3_sphere(self, other)

    def _intersect_plane(self, other):
        return _intersect_line3_plane(self, other)

    def connect(self, other):
        return other._connect_line3(self)

    def _connect_point3(self, other):
        return _connect_point3_line3(other, self)

    def _connect_line3(self, other):
        return _connect_line3_line3(other, self)

    def _connect_sphere(self, other):
        return _connect_sphere_line3(other, self)

    def _connect_plane(self, other):
        c = _connect_line3_plane(self, other)
        if c:
            return c

class Ray3(Line3):
    def __repr__(self):
        return 'Ray3(<%.2f, %.2f, %.2f> + u<%.2f, %.2f, %.2f>)' % \
            (self.p.x, self.p.y, self.p.z, self.v.x, self.v.y, self.v.z)

    def _u_in(self, u):
        return u >= 0.0

class LineSegment3(Line3):
    def __repr__(self):
        return 'LineSegment3(<%.2f, %.2f, %.2f> to <%.2f, %.2f, %.2f>)' % \
            (self.p.x, self.p.y, self.p.z,
             self.p.x + self.v.x, self.p.y + self.v.y, self.p.z + self.v.z)

    def _u_in(self, u):
        return u >= 0.0 and u <= 1.0

    def __abs__(self):
        return abs(self.v)

    def magnitude_squared(self):
        return self.v.magnitude_squared()

    def _swap(self):
        # used by connect methods to switch order of points
        self.p = self.p2
        self.v *= -1
        return self

    length = property(lambda self: abs(self.v))

class Sphere:
    __slots__ = ['c', 'r']

    def __init__(self, center, radius):
        assert isinstance(center, Vector3) and type(radius) == float
        self.c = center.copy()
        self.r = radius

    def __copy__(self):
        return self.__class__(self.c, self.r)

    copy = __copy__

    def __repr__(self):
        return 'Sphere(<%.2f, %.2f, %.2f>, radius=%.2f)' % \
            (self.c.x, self.c.y, self.c.z, self.r)

    def _apply_transform(self, t):
        self.c = t * self.c

    def intersect(self, other):
        return other._intersect_sphere(self)

    def _intersect_point3(self, other):
        return _intersect_point3_sphere(other, self)

    def _intersect_line3(self, other):
        return _intersect_line3_sphere(other, self)

    def connect(self, other):
        return other._connect_sphere(self)

    def _connect_point3(self, other):
        return _connect_point3_sphere(other, self)

    def _connect_line3(self, other):
        c = _connect_sphere_line3(self, other)
        if c:
            return c._swap()

    def _connect_sphere(self, other):
        return _connect_sphere_sphere(other, self)

    def _connect_plane(self, other):
        c = _connect_sphere_plane(self, other)
        if c:
            return c

class Plane:
    # n.p = k, where n is normal, p is point on plane, k is constant scalar
    __slots__ = ['n', 'k']

    def __init__(self, *args):
        if len(args) == 3:
            assert isinstance(args[0], Point3) and \
                   isinstance(args[1], Point3) and \
                   isinstance(args[2], Point3)
            self.n = (args[1] - args[0]).cross(args[2] - args[0])
            self.n.normalize()
            self.k = self.n.dot(args[0])
        elif len(args) == 2:
            if isinstance(args[0], Point3) and isinstance(args[1], Vector3):
                self.n = args[1].normalized()
                self.k = self.n.dot(args[0])
            elif isinstance(args[0], Vector3) and type(args[1]) == float:
                self.n = args[0].normalized()
                self.k = args[1]
            else:
                raise AttributeError( '%r' % (args,))

        else:
            raise AttributeError( '%r' % (args,))
        
        if not self.n:
            raise AttributeError( 'Points on plane are colinear')

    def __copy__(self):
        return self.__class__(self.n, self.k)

    copy = __copy__

    def __repr__(self):
        return 'Plane(<%.2f, %.2f, %.2f>.p = %.2f)' % \
            (self.n.x, self.n.y, self.n.z, self.k)

    def _get_point(self):
        # Return an arbitrary point on the plane
        if self.n.z:
            return Point3(0., 0., self.k / self.n.z)
        elif self.n.y:
            return Point3(0., self.k / self.n.y, 0.)
        else:
            return Point3(self.k / self.n.x, 0., 0.)

    def _apply_transform(self, t):
        p = t * self._get_point()
        self.n = t * self.n
        self.k = self.n.dot(p)

    def intersect(self, other):
        return other._intersect_plane(self)

    def _intersect_line3(self, other):
        return _intersect_line3_plane(other, self)

    def _intersect_plane(self, other):
        return _intersect_plane_plane(self, other)

    def connect(self, other):
        return other._connect_plane(self)

    def _connect_point3(self, other):
        return _connect_point3_plane(other, self)

    def _connect_line3(self, other):
        return _connect_line3_plane(other, self)

    def _connect_sphere(self, other):
        return _connect_sphere_plane(other, self)

    def _connect_plane(self, other):
        return _connect_plane_plane(other, self)




#===================================================================================================

#
# CLASS DEFINITIONS (SPECIFIC)
#



# CLASS bead
#
# 3 attributes class Vector3: endpoint 1 (endp1); centroid (centrd); endpoint 2 (endp2)
# 1 attribute type floating point: the bead size (ext)
#
class bead:
    def __init__(self,endp1=Vector3(0.,0.,0.),centrd=Vector3(0.,0.,0.),endp2=Vector3(0.,0.,0.),ext=DIA/2):
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
        return          str(self.endp1.x)+' '+ str(self.endp1.y)+' '+ str(self.endp1.z)+' '+str(self.ext)+'\n'+ \
                        str(self.centrd.x)+' '+ str(self.centrd.y)+' '+ str(self.centrd.z)+' '+str(self.ext)+'\n'+ \
                        str(self.endp2.x)+' '+ str(self.endp2.y)+' '+ str(self.endp2.z)+' '+str(self.ext)
#
#  Final (prints the centroid coordinates and the size)
    def finalstr(self):
        return          str(self.centrd.x)+' '+ str(self.centrd.y)+' '+ str(self.centrd.z)+' '+str(self.ext)
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
        d1=(self.endp1-self.centrd).magnitude()
        d2=(self.centrd-self.endp2).magnitude()
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
        self.endp1=Vector3(0.,0.,0.)





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
    def quaplanar(self,ibead,angle):
        axis=(self.bead[ibead].endp2-self.bead[ibead].endp1).cross(self.bead[ibead-1].endp2-self.bead[ibead-1].endp1)
        qua=Quaternion.new_rotate_axis(angle,axis.normalized()).normalize()
        return qua, axis
            
#
# Using quaternions, modifies the dihedral angle between the ibead-th and the (ibead-1)-th beads
    def quadihedral(self,ibead,angle):
        axis=self.bead[ibead-1].endp2-self.bead[ibead-1].endp1
        qua=Quaternion.new_rotate_axis(angle,axis.normalized()).normalize()
        return qua
            

#
# Random chain perturbation through a single bead (in this version, d1 and d2 are kept fixed)
    def perturb(self,ibead):
        ch=copy.deepcopy(self)
        deltap=(random.random()-0.5)*RANDPLA
        deltad=(random.random()-0.5)*RANDDIE
        [quap, axis]=ch.quaplanar(ibead,deltap)
        quad=ch.quadihedral(ibead,deltad)
        if axis.magnitude():
            for i in range(ibead,len(self)):
                a=ch.bead[i]
                a.zerobead()
                ch.bead[i].endp1=ch.bead[i-1].endp2
                ch.bead[i].centrd=quap*a.centrd+ch.bead[i-1].endp2
                ch.bead[i].endp2=quap*a.endp2+ch.bead[i-1].endp2
        else:
            for i in range(ibead,len(self)):
                a=ch.bead[i]
                a.zerobead()
                a.centrd=Vector3(a.centrd.x*math.cos(deltap)-a.centrd.y*math.sin(deltap),\
                                 a.centrd.x*math.sin(deltap)+a.centrd.y*math.cos(deltap),a.centrd.z)
                a.endp2=Vector3(a.endp2.x*math.cos(deltap)-a.endp2.y*math.sin(deltap),\
                                 a.endp2.x*math.sin(deltap)+a.endp2.y*math.cos(deltap),a.endp2.z)
                ch.bead[i].endp1=ch.bead[i-1].endp2
                ch.bead[i].centrd=a.centrd+ch.bead[i-1].endp2
                ch.bead[i].endp2=a.endp2+ch.bead[i-1].endp2
                
        for i in range(ibead,len(self)):
            b=ch.bead[i]
            b.zerobead()
            ch.bead[i].endp1=ch.bead[i-1].endp2
            ch.bead[i].centrd=quad*b.centrd+ch.bead[i-1].endp2
            ch.bead[i].endp2=quad*b.endp2+ch.bead[i-1].endp2

        return ch


#===================================================================================================

#
# FUNCTION DEFINITIONS:
#
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
#  - binning (bins a matrix given the sizes of the relevant diagonal blocks)
#  - compbead (computes an equivalent low-resolution bead given a reconstructed subchain)
#  - savepar (saves output parameters to file)
#  - saveconf (saves output configuration to file)
#  - constr (computes the initial beas sizes given the geometric parameters)
#  - inizchain (computes an initial-guess chain)
#  - align (alings a subchain to its lower-resolution bead counterpart)
#  - compose (reconstructs the full-resolution chain from the lower-resolution subchains)
#  - drawSphere (plots each bead as a sphere)
#  - set_axes_equal (equalizes axis scales in display)
#  - output (plots and saves the final configuration to file)
#  - chromstruct (overall recursive procedure)
#


# FUNCTION distmatrix
#
# Distance matrix given a chain configuration (distances between centroids; returns an upper-triangular matrix)
def distmatrix(conf):
    M=np.zeros([len(conf),len(conf)])
    for i in range(len(conf)):
        for j in range(i+1,len(conf)):
            M[i][j]=(conf.bead[j].centrd-conf.bead[i].centrd).magnitude()
    return M
             

# FUNCTION block (New version, after Mizuguchi et al. 2014)
#
# TAD detection (approximate diagonal block extraction from contact matrix)
def block(contacts,window=3,minsize=4,span=7):

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
    

# FUNCTION populate
#
# Returns a list of 2-tuples specifying the bead pairs with relevant contact numbers 
def populate(mat):
    string= "block size "+str(len(mat))+'\n'
    print string
    outdata.write(string)
    matrix=mat-np.diag(np.diag(mat)) # Delete the diagonals to be neglected
    for ineg in range(1,diagneg): 
        matrix=matrix-np.diag(np.diag(mat,ineg),ineg)
    L=[]
    # This strategy selects a fraction datafact of the total number of pairs in the block
    c=sorted(matrix.reshape([len(matrix)**2]),reverse=True)
    for i in range(len(matrix)):
        for j in range(i+diagneg,len(matrix)):
            if matrix[i,j]>=c[int(0.5*len(matrix)*(len(matrix)-1)*datafact)]: 
                L.append((i,j))
    string = "number of relevant pairs in this block: "+str(len(L))+str("(datafact = "+str(datafact)+")\n")
    print string
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
        T2=incrtem*T
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
def fitcomp(M,n,pairs,conf,matrix,p=1):
    fit=0. 
    for k in range(len(pairs)):
#        fit=fit+matrix[pairs[k]]*M[pairs[k]]
        fit=fit+matrix[pairs[k]]*(M[pairs[k]]-conf.bead[pairs[k][0]].ext-conf.bead[pairs[k][1]].ext)**2
    fit=fit/LMAX                    # fitness normalized to LMAX
    if not p:
        print "data fit   ",fit     # p = print flag (default 1: no print)
        outdata.write("data fit   "+str(fit)+'\n')
    return fit

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
        print "constraint ",constr   # p = print flag (default 1: no print)
        outdata.write("constraint "+str(constr)+'\n')
    return constr

# FUNCTION energy
#
# Sums the weighted constraint part to the data fit part
def energy(pairs,conf,matrix,lamb=0.,p=1):
    n=len(conf)
    M=distmatrix(conf)
    fit=fitcomp(M,n,pairs,conf,matrix,p=1)
    constr=constrcomp(M,n,conf,p=1)        # constr is the constraint part of the energy, 
                                            # not yet multiplied by parameter lamb
    constr=lamb*constr
    energ=fit+constr 
    return energ,fit,constr



#
# SIMULATED ANNEALING
#
# FUNCTION annealing
#
# Sets the regularization parameter, sets the start temperature, samples the energy distribution
def annealing(pairs,matrix,chain):
    n=len(matrix)
    conf=copy.deepcopy(chain)
    conf.zerochain()
    m=len(conf)

    # Computes a regularization parameter from a fixed relative weight of the constraint part
    fitlist=[]
    constrlist=[]
    for j in range(avgenergy):
        for i in range(1,n):
            conflamb=conf.perturb(i)
            M=distmatrix(conflamb)
            fitlamb=fitcomp(M,m,pairs,conflamb,matrix)
            constrlamb=constrcomp(M,m,conflamb)
            conf=conflamb
            fitlist.append(fitlamb)
            constrlist.append(constrlamb)
    fitlist.sort()
    constrlist.sort()
    meanfit=np.mean(fitlist[0:-len(fitlist)/(100-percenergy)])
    meanconstr=np.mean(constrlist[0:-len(constrlist)/(100-percenergy)])
    lamb=meanfit/meanconstr*regulenergy
    print "lambda", lamb,'(average within '+str(int(percenergy))+'-th percentile)\n'
    outdata.write("lambda "+str(lamb)+'average within '+str(int(percenergy))+'-th percentile)\n')

#
#  Added to start from initial configuration (comment out if needed)
#
    conf=copy.deepcopy(chain)
    conf.zerochain()
##
    [Phi,fit,constr]=energy(pairs,conf,matrix,lamb) # Initial energy

        
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
            [Phistar,fitstar,constrstar]=energy(pairs,confstar,matrix,lamb)
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
                print "T = ", T
                outdata.write("T = "+str(T)+'\n')
                print "Acceptance rate",accrate
                outdata.write("Acceptance rate "+str(accrate)+'\n')
                T=warm(T,accrate)
                naccept=0
                nincr=0
                nincracc=0
    string = "\nAnnealing - warm up\n# cycles "+str(iterat)\
             +"\nAcceptance rate "+str(accrate)\
             +"\nStart temperature "+str(T)+'\n\n'\
             +"Annealing - sampling\n"

    print string,
    outdata.write(string)

    # annealing phase
#
#  Added to start from initial configuration (uncomment if needed)
#
#    conf=copy.deepcopy(chain)
    conf.zerochain()
    [Phi,fit,constr]=energy(pairs,conf,matrix,lamb) # Initial energy
    string = "\nStart energy "+str(Phi)\
             +"\nData fit "+str(fit)+" Constraint "+str(constr)+'\n\n'
    print string,
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
            [Phistar,fit,constr]=energy(pairs,confstar,matrix,lamb,1)
            if checktest(Phi,Phistar,T):
                naccept=naccept+1
                Taccept.append(T)
                Phiaccept.append([fit,constr])
                iteraccept.append((iterat-1)*n+i+1)
                conf=confstar
                if Phistar > Phi:
                    nincrease=nincrease+1
                Phi=Phistar
        if not iterat%1000:  # print values every 1000 cycles
            if len(Phiaccept)>0:
                string = "T = "+str(T)\
                         +"\nenergy "+str(Phi)+" ("+str(Phiaccept[-1][0])+" + "+str(Phiaccept[-1][1])+")\n"
            else:
                string="T = "+str(T)+". No transition accepted in 1000 cycles."
            print string, 
            outdata.write(string)
        T=cool(T)

    string = "\nAnnealing - Final temperature "+str(T)\
             +"\nFinal energy "+str(Phi)\
             +"\nTotal # cycles "+str(iterat)\
             +"\nTotal accepted updates "+str(naccept)\
             +"\nUpdates increasing energy "+str(nincrease)+'\n'
    print string,
    outdata.write(string)

    Phiaccept=np.array(Phiaccept)
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
        iinf=np.sum(sizes[0:i],dtype=np.uint64)
        for j in range(i,n):
            jinf=np.sum(sizes[0:j],dtype=np.uint64)
            binnedmatrix[i,j]=np.sum(matrix[iinf:iinf+sizes[i],jinf:jinf+sizes[j]])
    return binnedmatrix
    




# FUNCTION compbead
#
# Computes a lower-resolution bead from a reconstructed subchain
def compbead(conf):
    bigbead=bead()
    # centroid
    bigbead.centrd.x=np.sum(conf.bead[i].centrd.x for i in range(len(conf)))/len(conf)
    bigbead.centrd.y=np.sum(conf.bead[i].centrd.y for i in range(len(conf)))/len(conf)
    bigbead.centrd.z=np.sum(conf.bead[i].centrd.z for i in range(len(conf)))/len(conf)
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
    bigbead.endp1=Vector3(0.,0.,0.)
    bigbead.centrd=Vector3(d1,0.,0.)
    bigbead.endp2=Vector3(d1-d2*math.cos(ph),d2*math.sin(ph),0.)
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
def constr(n,NMAX):
    L=LMAX-(LMAX-LMIN)*n/NMAX
    bead1=bead(Vector3(0.,0.,0.),Vector3(L/2.,0.,0.),Vector3(L,0.,0.),ext=L/2.)
    return bead1
    
    


# FUNCTION inizchain
#
# Returns the start configuration at level 0
def inizchain(hm,NMAX=0):
    if not NMAX: NMAX=hm.max() # NMAX=0 means that the maximum frequency in the current data matrix is assumed
    chain1=chain([])
    for i in range(len(hm)):
        chain1.chainadd(constr(hm[i,i],NMAX)) 
    chain1.formchain()
    return chain1




# FUNCTION align
#
# Aligns a subchain to its low-resolution counterpart (rotated low-resolution bead at the successive level)
def align(cha,bd):
    ch=copy.deepcopy(cha)
    ch.zerochain() # Place the subchain in the origin
    # basis vectors for subchain
    v2=Vector3(0.,0.,0.)
    v2.x=np.sum(ch.bead[i].centrd.x for i in range(len(ch)))*1./len(ch)
    v2.y=np.sum(ch.bead[i].centrd.y for i in range(len(ch)))*1./len(ch)
    v2.z=np.sum(ch.bead[i].centrd.z for i in range(len(ch)))*1./len(ch)
    v3=ch.bead[-1].endp2
    v1=np.cross(v2,v3)
    v1=Vector3(v1[0],v1[1],v1[2])
    base1=[v1.normalized(),v2.normalized(),v3.normalized()]

    bd.zerobead() # Place the low-resolution bead in the origin

    # basis vectors for bead
    w2=bd.centrd
    w3=bd.endp2
    w1=np.cross(w2,w3)
    w1=Vector3(w1[0],w1[1],w1[2])
    base2=[w1.normalized(),w2.normalized(),w3.normalized()]

    # transform base1 into base2 (Rotation matrix)
    M1=np.matrix(base1)
    M2=np.matrix(base2)
    M1inv=LA.inv(M1)
    Lin=M1inv*M2
    for j in range(len(ch)):
        ch.bead[j].zerobead()
        b=np.matrix(ch.bead[j].centrd)*Lin
        ch.bead[j].centrd=Vector3(b[0,0],b[0,1],b[0,2])
        b=np.matrix(ch.bead[j].endp2)*Lin
        ch.bead[j].endp2=Vector3(b[0,0],b[0,1],b[0,2])
    return ch
     




# FUNCTION compose
#
# Recursive procedure to reconstruct the full-resolution chain from the lower-resolution versions
def compose(conf,level):
    Csuc=chain([])    # Initialize the higher-resolution chain
    level=level-1     # Raise level
    for ibead in range(len(conf)):  # Scan the beads at the current resolution
        mat=np.loadtxt(filen+'_'+timemark+'_'+str(level)+'_'+str(ibead)+'.txt') # read partial data
        subchain=chain([])
        for i in range(len(mat)/3):                # form the subchain for the current bead
            subchain.chainadd(bead(Vector3(mat[3*i,0],mat[3*i,1],mat[3*i,2]),\
                                   Vector3(mat[3*i+1,0],mat[3*i+1,1],mat[3*i+1,2]),\
                                   Vector3(mat[3*i+2,0],mat[3*i+2,1],mat[3*i+2,2]),mat[3*i,3]))
        ch=align(subchain,conf.bead[ibead])       # Align the subchain with the current bead
        Csuc.chainmerge(ch)                        # Append the new subchain to the higher-resolution chain
    Csuc.formchain()                               # Form the higher-resolution chain

    if level > 0:                                  # Pass to successive level, if any
        Csuc=compose(Csuc,level)
        return Csuc
    else:
        return Csuc                                # Full-resolution chain





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
    ax.set_aspect('equal')
    col_gen=cycle('bgrcmyk')
    xl=[]
    yl=[]
    zl=[]
    for i in range(len(conf)):
        x=conf.bead[i].centrd.x
        y=conf.bead[i].centrd.y
        z=conf.bead[i].centrd.z
        xl.append(conf.bead[i].centrd.x)
        yl.append(conf.bead[i].centrd.y)
        zl.append(conf.bead[i].centrd.z)
        r=conf.bead[i].ext

## Display spherical beads (uncomment if needed)
##        (xs,ys,zs) = drawSphere(x,y,z,r)
##        ax.plot_wireframe(xs, ys, zs, color=col_gen.next())    

    ax.plot(xl, yl, zl,'r',linewidth=3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    set_axes_equal(ax)
    plt.show()





# FUNCTION chromstruct
#
# Recursive procedure to reconstruct the full-resolution chain
def chromstruct(contacts,chaininf,level=0):
    [blocks,sizes,bounds]=block(contacts,minsize=minsize,window=window,span=span) # extracts the TADs at the current level
    n=len(sizes)
    out=open(filen+'_'+timemark+'_'+str(level)+'_BlockSizes.txt',"w")
    for i in range(n):
        out.write(str(sizes[i])+'\n')
    out.close()

    string=       "\n\n========================================\nCHROMSTRUCT ver "+version\
                  +"\n========================================"\
                  +"\n level "+str(level)\
                  +"\n# Blocks "+str(n)\
                  +"\n(smoothing window "+str(window)+"; averaging span "+str(span)+"; minimum block size "+str(minsize)+")"\
                  +"\n# Neglected diagonals "+str(diagneg)\
                  +"\nEnergy: relative weight, constraint part: "+str(regulenergy)\
                  +"\nEnergy: neighbor interaction function: scale = "+str(energy_scale)+"; exponent = "+str(energy_exp)\
                  +"\nAssumed bead sizes: max sigma * "+str(extrate)\
                  +"\nStop energy relative tolerance: "+str(StopTolerance)+" over "+str(itstop)+" cycles\n"
    print string,
    outdata.write(string)
   
    newchain=chain([])

    chainlist=[]
    for i in range(1,n+1):         # Form a list of subchains at the current level
        chainlist.append(chain(chaininf.bead[bounds[i-1]:bounds[i]]))
    
    for i in range(n):             # Scan the subchains (intrinsically parallelizable)
        string = "\n\n========================================\nlevel "+str(level)+" block "+str(i)+'\n'
        print string,
        outdata.write(string)     
        
        L=populate(blocks[i]) # select the pairs for the data fit part
        string = "Relevant pairs "+str(L)+'\n\n'
        print string,
        outdata.write(string)
        

        [C, Tseries, Phiseries, iteraccept] = annealing(L,blocks[i],chainlist[i]) # Sample the current subchain

        #
        # (Optional) Plot data fit and constraint in the current cycle
        #

        if display[0] == 'y':
            pylab.plot(Phiseries[:,0],label='Data score')
            pylab.plot(Phiseries[:,1],label='Constraint score')
            pylab.plot(Phiseries[:,0]+Phiseries[:,1],label='Total score')
            plt.xlabel("Accepted updates")
            pylab.legend()
            pylab.show()

            # Plot accepted vs proposed updates in the current cycle
            pylab.plot(iteraccept,range(len(iteraccept)))
            plt.ylabel('Accepted update')
            plt.xlabel('Proposed update')
            pylab.show()

        # end (Optional)
        #
        
        savepar(Phiseries, level, i)                  # Save the features of the annealing schedule
        saveconf(C,level,i)                           # Save current subchain configuration
        newbead=compbead(C)                           # Compute lower-resolution bead from current subchain
        newchain.chainadd(copy.deepcopy(newbead))     # Append lower-res bead to lower-res chain
        
    if n>1:                                           # Increase level if appropriate
        level=level+1
        newchain.formchain()
        contacts=binning(sizes,contacts)              # Bin data matrix to successive resolution

        chromstruct(contacts,newchain,level)          # call procedure at the successive level
    else:
        if level > 0:
            Clast=compose(C,level)                    # recompose full-resolution chain
        else:
            Clast=C
        string = "CPU time (secs) "+str(time.clock())+'\n'
        print string,
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
##
##    
##NC=RIS/NB             # # fragments per genomic resolution unit
##LMAX=NC*DIA/np.pi     # hypothesized maximum size of a bead in the full-resolution model (nm)
##LMIN=NC**(1./3)*math.sqrt(3)*DIA # hypothesized minimum bead size in the full-resolution model (nm)
##NMAX=                 # Maximum contact frequency: fixed or computed in the current data matrix (see function inizchain)
##extrate=              # Size tuning for beads at levels > 0
##
###==================================================================================================
###

string=       "\n\n========================================\nCHROMSTRUCT ver "+version\
              +"\nCopyright (C) 2016 Emanuele Salerno, Claudia Caudai\n\n"\
              +"This program comes with ABSOLUTELY NO WARRANTY; for details type `sw'.\n"\
              +"This is free software, and you are welcome to redistribute it\n"\
              +"under certain conditions; type `sc' for details.\n"
print string
opt=raw_input("type 'sw' for warranty, 'sc' for conditions, or <enter> to continue\n\n")
if opt=="sw":
    print warranty
    opt2=raw_input("type 'sc' for conditions, or <enter> to continue\n\n")
    if opt2=="sc": print conditions
elif opt=="sc":
    print conditions
    opt2=raw_input("type 'sw' for warranty, or <enter> to continue\n\n")
    if opt2=="sw": print warranty


# timestamp to identify all the partial result files in the current run
t=time.localtime()
timemark=str(t.tm_year)[2:4]+"-%02d-%02d-%02d%02d" % (t.tm_mon,t.tm_mday,t.tm_hour,t.tm_min)



filen=raw_input('File name (.txt) = ? ')      # Read data file name

try: hm=np.loadtxt(filen+'.txt')              # Read contact data
except:
    try: hm=np.loadtxt(filen)
    except:
        print 'No such file: no data read\n'
        sys.exit(0)

if filen[-4:len(filen)]=='.txt':
    filen=filen[0:-4]

outdata=open(filen+'_'+timemark+'_Log.txt',"w") # Open logfile

if NMAX:
    string="\n\nFIXED NMAX = "+str(NMAX)
    print string
    outdata.write(string)

conf=inizchain(hm,NMAX)                          # Set initial guess

for i in range(len(hm)):                         # Set the lower triangle of the contact matrix to zero
    for j in range(i):
        hm[i,j]=0.

display=raw_input("Do you need intermediate plots? (y/n) ")
if display=='':display='n'

chromstruct(hm,conf)                             # Call recursion

outdata.close()                                  # Close logfile


