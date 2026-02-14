import pathlib
from src.OptimizationToolkit import *
from src.RocketSimToolkit import *

F_15Motor=Motor(filepath='MotorData/Estes_F15.csv')

controlRocket=Rocket(drymass=0.3,#Should be changed
                     initVel=Vector(0,0,0.2),#Simulates launch rod
                     initPos=Vector(0,0,40*1000),
                     motor=F_15Motor)

controlDrag=0.5 #Needs to be simulated, at 50m/s

iteration=Iteration(str(pathlib.Path(__file__).resolve().parent), input("Start from Iteration: ")).nextIteration()

print(iteration.rows)

def calcNoseMass(control,row):
    return control["noseMass"]*row["SA"]/control["SA"]

iteration.calculateNulls("noseMass",calcNoseMass)

def calculateAltitude(control,row):
    rocket=controlRocket
    rocket.drymass=controlRocket.drymass-control["noseMass"]+row["noseMass"]
    rocket.AC_dFromDrag(controlDrag-control["drag"]+row["drag"],earthAtmosphericModel(controlRocket.pos.z)[0])

    deltaT=0.001#shorter timestep for motor burn
    while(rocket.vel.z>=0):#Only simulate upwards flight
        rocket.simStep(deltaT)
        if(rocket.mass==rocket.drymass):
            deltaT=0.02 #Longer timesteps after motor burnout
        print(rocket)
        return rocket.pos.z-controlRocket.pos.z#apoapsis-initial altitude
    
iteration.calculateNulls("Altitude", calculateAltitude)