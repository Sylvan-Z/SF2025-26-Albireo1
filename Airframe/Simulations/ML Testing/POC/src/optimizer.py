import keras
import numpy as np
import itertools

filepath = input("Filepath: ")

model = keras.models.load_model(filepath+"/model.keras")

searchStep = 0.01

allFeatures = list(map(list, itertools.product(list(np.arange(0, 1+searchStep, searchStep)), repeat=2)))
allDrags = list(map(lambda x: x[0], model.predict(np.array(allFeatures))))

bestFeatures=[]
bestdrag=float("inf")

for i in range(len(allFeatures)):
    if allDrags[i]<bestdrag:
        bestdrag=allDrags[i]
        bestFeatures=allFeatures[i]

print("Done broad search")
print(format("Predicted %s as the best with drag %f" % (str(bestFeatures), bestdrag)))
open(filepath+"/log.txt","w+").write(format("Predicted %s as the best with drag %f\n Actual drag: " % (str(bestFeatures), bestdrag)))
print("Starting gradient descent refinement")

deltaF=0.0001
deltaT=0.0001
derivitives=[float("inf")]
iterations=0
while sum(map(lambda x: abs(x), derivitives))/len(derivitives)>0.0001:
    derivitives=[]
    for i in range(len(bestFeatures)):
        dFeatures=bestFeatures
        dFeatures[i]+=deltaF
        dDrag=model.predict(np.array([dFeatures]))[0][0]
        derivitives.append((dDrag-bestdrag)/deltaF)
    for i in range(len(bestFeatures)):
        bestFeatures[i]-=derivitives[i]*deltaT
        bestFeatures[i]=np.clip(bestFeatures[i],0,1)
    bestdrag=model.predict(np.array([bestFeatures]))[0][0]
    iterations+=1
    print("Iteration %d: Predicted %s as the best with drag %f, average gradient %f" % (iterations, str(bestFeatures), bestdrag, sum(derivitives)/len(derivitives)))

print("Done refinement")
print(format("Predicted %s as the best with drag %f" % (str(bestFeatures), bestdrag)))