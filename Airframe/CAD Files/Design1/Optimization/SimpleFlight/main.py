from src.RocketSimToolkit import *

F_15Motor=Motor(filepath='MotorData/Estes_F15.csv')

rocket=Rocket(drymass=0.200,
                     initVel=Vector(0,0,1),#Simulates launch rod
                     initPos=Vector(0,0,40*1000),
                     motor=F_15Motor)

rocket.AC_dFromDrag(1.82844539158995,50,earthAtmosphericModel(0)[0])

deltaT=0.001

printInt=0.1

print(rocket)
print('-----')

while(rocket.vel.z>=0):
    rocket.simStep(deltaT)
    if(rocket.t%printInt<deltaT):
        print(rocket)
        print('-----')
    if(rocket.mass==rocket.drymass):
        deltaT=0.1

print(rocket)