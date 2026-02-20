import math
import csv as csvReaders
import numpy as np

from copy import copy
from typing import Callable

class Vector:
    def __init__(self, *args:float):
        self.components = tuple(args)
        self.dimensions = len(self.components)
        self.generateXYZ()

    def __getitem__(self,i):
        return self.components[i]

    def generateXYZ(self):
        if(self.dimensions>=1):self.x=self[0]
        if(self.dimensions>=2):self.y=self[1]
        if(self.dimensions>=3):self.z=self[2]
    
    def mag(self):
        return math.sqrt(sum(map(lambda x:x**2,self.components)))
    
    def __add__(self, other:'Vector'):
        return Vector(*tuple([self[i]+other[i] for i in range(self.dimensions)]))
    
    def __sub__(self, other:'Vector'):
        return Vector(*tuple([self[i]-other[i] for i in range(self.dimensions)]))
    
    def __mul__(self, factor:float):
        return Vector(*tuple(map(lambda x:factor*x,self.components)))
    
    def __rmul__(self, factor:float):
        return Vector(*tuple(map(lambda x:factor*x,self.components)))
    
    def __truediv__(self, factor:float):
        return self*(1/factor)
    
    def unitVector(self):
        return self/self.mag()
    
    def __repr__(self):
        return str(list(self.components))
    
def earthAtmosphericModel(h):
    #If statement for piecewise function, from https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html
    if h > 25000:
        T=-131.21+0.00299*h
        p=2.488*math.pow((T+273.1)/216,-11.388)
    elif 11000<h<=25000:
        T=-56.46
        p=22.65*math.pow(math.e,1.73-0.000157*h)
    elif h<=11000:
        T=15.04-0.00649*h
        p=101.29*math.pow((T+273.1)/288.08,5.256)
    rho=p/(0.2869*(T+273.1))
    
    #Outputs in kg/m^3, °C, and kPa
    return rho, T, p

class Motor:
    def __init__(self, time:list[float]=[], force:list[float]=[], mass:list[float]=[], filepath:str=None):
        if(filepath!=None):
            csv=open(filepath,'r')
            reader=csvReaders.DictReader(csv)
            force=[]
            time=[]
            mass=[]
            for row in reader:
                force.append(float(row["f_t"]))
                time.append(float(row["t"]))
                mass.append(float(row["m"])/1000)

        self.force=np.array(force,dtype=float)
        self.time=np.array(time,dtype=float)
        self.mass=np.array(mass,dtype=float)

    def getMass(self,t:float)->float:
        return np.interp([t],self.time,self.mass)[0]
    
    def getThrust(self,t:float)->float:
        return np.interp([t],self.time,self.force)[0]

#Force Types
#Statics
@staticmethod
def gravity(rocket:'Rocket'):
    '''Calculates gravity, varying with the rocket's altitude'''
    g = -9.81   # m/s^2, gravity
    r = 6371000 #Earth mean radius, m
    return Vector(0,0,rocket.mass*g*(r/(r+rocket.pos.z))**2),0

@staticmethod
def thrust(rocket:'Rocket'):
    return rocket.vel.unitVector()*rocket.motor.getThrust(rocket.midT),0

@staticmethod
def drag(rocket:'Rocket'):
    '''Calculates linear drag coeffecient, varying rocket velocity'''
    v=rocket.vel.mag()
    rho,T,p=earthAtmosphericModel(rocket.pos.z)
    return Vector(0,0,0),-0.5*rho*rocket.AC_d*v #one V, because linear scaling by vector velocity


class Rocket:
    '''mass[kg]'''
    def __init__(self, drymass:float=0.5, AC_d=0.5, motor:Motor=Motor([0],[0],[0]), initPos:Vector=Vector(0,0,0), initVel:Vector=Vector(0,0,0), forces:list[Callable[['Rocket'],tuple[Vector,float]]]=[gravity,thrust,drag]):
        self.drymass=drymass
        self.AC_d=AC_d #Area-drag Coefficient, m^2

        self.pos:Vector=initPos
        self.vel:Vector=initVel

        self.motor=motor

        self.forces=forces

        self.t=0.0

        self.midT=0.0
        self.updateMassForces(0)

    def AC_dFromDrag(self, drag:float, v, rho):
        '''
        Args:
            drag: Recorded drag, newtons
            v: Relative airspeed, m/s
            rho: Air density, kg/m^3
        '''
        self.AC_d=2*drag/(rho*(v**2))

    def calculateForces(self):
        '''
        Returns: static forces, linear drag forces
        '''
        statForce=Vector(0,0,0)
        linForceFactor=0
        for forceFunc in self.forces:
            force=forceFunc(self)
            statForce+=force[0]
            linForceFactor+=force[1]

        return statForce, linForceFactor
    
    def calculateMass(self):
        return self.drymass+self.motor.getMass(self.midT)

    def updateMassForces(self,deltaT):
        self.midT=self.t+deltaT/2 #Store midpoint of simulated timestep for motor simulation
        self.mass=self.calculateMass()
        self.staticF,self.linF=self.calculateForces()

    def simStep(self, deltaT:float):
        self.updateMassForces(deltaT)
        #Create variables for easier manipulation
        m=self.mass
        F=self.staticF
        k=self.linF
        t=deltaT
        v_0=self.vel
        d_0=self.pos
        #When k is very small, only apply static forces to avoid floating point errors
        if(abs(k)<0.0001):
            #Calculate new velocity
            self.vel=((v_0*k+F)*t)/m+v_0
            #Calculate new position
            self.pos=((v_0*k+F)*(t**2))/(2*m)+(v_0*t)+d_0
        else:
            #Calculate new velocity
            self.vel=(math.exp(k*t/m)*(k*v_0+F)-F)/k
            #Calculate new position
            self.pos=((m*(math.exp(k*t/m)-1)*(k*v_0+F))/(k**2))-(F*t/k)+d_0
            #See Logbook day Feb.13 for derivitation of these formulea
        self.t+=deltaT

    def __repr__(self):
        return "\n".join([
            f"Time Since Launch: {self.t}",
            f"Velocity: {self.vel.mag()}",
            f"Velocity (Components): {str(self.vel)}",
            f"Position (Components): {str(self.pos)}",
            f"Mass: {str(self.mass)}",
            f"Motor Mass: {str(self.motor.getMass(self.t))}",
            f"Motor Thrust: {str(self.motor.getThrust(self.t))}",
            f"Static Force Components: {str(self.staticF)}",
            f"Linear Force Factor: {self.linF}",
            f"Linear Force Components: {self.vel*self.linF}",
            f"Total Forces: {self.vel*self.linF+self.staticF}",
            f"Total Acceleration: {(self.vel*self.linF+self.staticF)/self.mass}"
        ])
    
    def copy(self):
        return copy(self)