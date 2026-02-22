import pathlib
from src.OptimizationToolkit import *
from src.RocketSimToolkit import *

F_15Motor=Motor(filepath='MotorData/Estes_F15.csv')

controlRocket=Rocket(drymass=0.25,#Approx
                     initVel=Vector(0,0,1),#Simulates launch rod
                     initPos=Vector(0,0,40*1000),
                     motor=F_15Motor)

controlDrag=6.49716759127639 #Needs to be simulated, at 100m/s

#Null handlers
def calcMass(df, control):
    skinDensity=1.87/1374.137061 #g/mm^2
    volumeDensity=0.75/14721.515087 #g/mm^3
    return df["SA"]*skinDensity+df["V"]*volumeDensity

def calculateAltitude(df,control):
    rocket=controlRocket.copy()
    rocket.drymass=controlRocket.drymass-control["mass"]/1000+df["mass"]/1000
    rocket.AC_dFromDrag(controlDrag-control["drag"]+df["drag"], 100, earthAtmosphericModel(controlRocket.pos.z)[0])

    deltaT=0.001#shorter timestep for motor burn
    while(rocket.vel.z>=0):#Only simulate upwards flight
        rocket.simStep(deltaT)
        if(rocket.mass==rocket.drymass):
            deltaT=0.02 #Longer timesteps after motor burnout
    print(rocket)
    return rocket.pos.z-controlRocket.pos.z#apoapsis-initial altitude


while True:
    n=int(input("Start from Iteration (-1 to exit): "))

    if(n==-1):
        break

    iteration=Iteration(str(pathlib.Path(__file__).resolve().parent), n).nextIteration()

    print("Loaded dataframe, calculating nulls")
    print(iteration.df)

    iteration.calculateNulls("mass",calcMass)
        
    iteration.calculateNulls("Altitude", calculateAltitude)

    print("final DF")
    print(iteration.df)

    print("All nulls replaced, saving csv and starting optimization")

    iteration.saveAll()

    iteration.generateDeltas()

    print("Deltas:")

    print(iteration.deltas)

    xSets=[[iteration.df.at[i,'l'],iteration.df.at[i,'a'],iteration.df.at[i,'b']] for i in range(len(iteration.df))]
    ySets=[iteration.deltas.at[i,'Altitude']/5 for i in range(len(iteration.deltas))]#Factors are arbutrary, but keep i/o in a reasonable magnitude

    print(xSets)
    print(ySets)

    iteration.fit(xSets,ySets)

    optimizedX=iteration.optimize(True,np.arange(0.1,5,0.1),np.arange(0,1,0.05),np.arange(0,1,0.05))

    print(optimizedX)

    iteration.addRow({"l":optimizedX[0],"a":optimizedX[1],"b":optimizedX[2]})

    if input("Save Model?(Y/N): ")=="Y":
        iteration.saveAll()