from src.RocketSimToolkit import *

B4motor = Motor(filepath='MotorData/Estes_B4.csv') 

rocket = Rocket(drymass=0.061, motor=B4motor, initVel=Vector(0,0,1))
rocket.AC_dFromDrag(00.537343891398821, 50, earthAtmosphericModel(0)[0])
print(rocket.AC_d)

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