import keras
import numpy as np
import itertools

filepath = input("Filepath: ")

model = keras.models.load_model(filepath+"/model.keras")

controlDrag = None
with open(filepath+"/dataset.csv", "r") as inputFile:
    controlDrag = float(inputFile.readlines()[1].strip().split(",")[-1])
    inputFile.close()

searchStep = 0.1

maxes=[9,9,1.65]

allFeatures = list(map(list, itertools.product(list(np.arange(0, maxes[0]+searchStep, searchStep)),list(np.arange(0, maxes[1]+searchStep, searchStep)),list(np.arange(0, maxes[2]+searchStep, searchStep)))))
allDrags = list(map(lambda x: x[0]+controlDrag, model.predict(np.array(allFeatures))))

bestFeatures=[]
bestdrag=float("inf")

for i in range(len(allFeatures)):
    if allDrags[i]<bestdrag:
        bestdrag=allDrags[i]
        bestFeatures=allFeatures[i]
        print(format("New best drag: Predicted %s as the best with drag %f" % (str(bestFeatures), bestdrag)))

print("Done broad search")
print(format("Predicted %s as the best with drag %f" % (str(bestFeatures), bestdrag)))
print("Starting gradient descent refinement")

deltaF=0.001
deltaT=0.0005
residuals=[float("inf")]
iterations=0
while sum(map(lambda x: abs(x), residuals))/len(residuals)>0.00003:
    allDFeatures=[]
    for i in range(len(bestFeatures)):
        allDFeatures.append(bestFeatures.copy())
        allDFeatures[-1][i]+=deltaF
    derivitives=list(map(lambda x:(bestdrag-x[0]-controlDrag)/deltaF, model.predict(np.array(allDFeatures))))
    newFeatures=[]
    residuals=[]
    for i in range(len(bestFeatures)):
        newFeatures.append(bestFeatures[i]+derivitives[i]*deltaT)
        newFeatures[-1]=np.clip(newFeatures[-1],0,maxes[i])
        residuals.append(newFeatures[-1]-bestFeatures[i])
    bestFeatures=newFeatures.copy()
    bestdrag=model.predict(np.array([bestFeatures]))[0][0]+controlDrag
    iterations+=1
    print("Iteration %d: Predicted %s as the best with drag %f, average residual %f" % (iterations, str(bestFeatures), bestdrag, sum(map(lambda x: abs(x), residuals))/len(residuals)))

print("Done refinement")
print(format("Predicted %s as the best with drag %f" % (str(bestFeatures), bestdrag)))
open(filepath+"/log.txt","w+").write(format("Predicted %s as the best with drag %f\nActual drag: " % (str(bestFeatures), bestdrag)))
open(filepath+"/dataset.csv","a").write(format("\n%s," % (",".join(list(map(str, bestFeatures))))))