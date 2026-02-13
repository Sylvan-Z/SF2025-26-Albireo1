import math

class Vector:
    def __init__(self, components:tuple[float]):
        self.components = components
        self.dimensions = len(components)
        self.generateXYZ()

    def __init__(self, *args:float):
        self.components = tuple(args)
        self.dimensions = len(self.components)
        self.generateXYZ()

    def generateXYZ(self):
        if(self.dimensions>=1):self.x=self.components[0]
        if(self.dimensions>=2):self.y=self.components[1]
        if(self.dimensions>=3):self.z=self.components[2]
    
    def mag(self):
        return Vector(math.sqrt(sum(map(lambda x:x**2,self.components))))
    
    def __add__(self, other:'Vector'):
        return Vector(tuple([self[i]+other[i] for i in range(self.dimensions)]))
    
    def __sub__(self, other:'Vector'):
        return Vector(tuple([self[i]-other[i] for i in range(self.dimensions)]))
    
    def __mul__(self, factor:float):
        return Vector(tuple(map(lambda x:factor*x,self.components)))
    
    def __div__(self, factor:float):
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

#Force Types
#Statics
@staticmethod
def gravity(rocket:'Rocket'):
    '''Calculates gravity, varying with the rocket's altitude'''
    g = -9.81   # m/s^2, gravity
    r = 6371000 #Earth mean radius, m
    return Vector(0,0,rocket.mass*g*r/((r+rocket.pos.z)**2))


def drag(rocket:'Rocket'):
    '''Calculates gravity, varying with the rocket's altitude'''
    v=rocket.vel.mag()
    rho,T,p=earthAtmosphericModel(rocket.pos.z)
    return -0.5*rho*rocket.AC_d*v #one V, because linear scaling by vector velocity



class Rocket:
    '''mass[kg]'''
    def __init__(self, mass=1, AC_d=0.5, initPos:Vector=Vector(0,0,0), initVel:Vector=Vector(0,0,0), statForces:list[function]=[gravity], linForces:list[function]=[drag]):
        self.mass=mass
        self.AC_d=AC_d #Area-drag Coefficient, m^2

        self.pos:Vector=initPos
        self.vel:Vector=initVel

        self.statForceFunctions=statForces
        self.linForceFunctions=linForces

        self.t=0.0

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
        for force in self.statForceFunctions:
            statForce+=force(self)

        linForceFactor=0
        for force in self.linForceFunctions:
            linForceFactor+=force(self)

        return statForce, linForceFactor

    def simStep(self, deltaT:float):
        #Create variables for better manipulation
        F,k=self.calculateForces()
        m=self.mass
        t=deltaT
        v_0=self.vel
        d_0=self.pos
        #When k is very small, only apply static forces to avoid floating point errors
        if(k<0.001):
            #Calculate new velocity
            self.vel=(F*t)/m+v_0
            #Calculate new position
            self.pos=(F*(t**2))/(2*m)+(v_0*t)+d_0
        else:
            #Calculate new velocity
            self.vel=(math.exp(k*t/m)*(k*v_0+F)-F)/k
            #Calculate new position
            self.pos=((m*(math.exp(k*t/m)-1)*(k*v_0+F))/(k**2))-(F*t/k)+d_0