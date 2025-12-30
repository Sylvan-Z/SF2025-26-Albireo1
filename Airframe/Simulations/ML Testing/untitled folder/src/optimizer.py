import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy

filepath = input("Filepath: ")

step = float(input("Step size: "))

model=keras.models.load_model(filepath+"/model.keras")

besta=0
bestb=0
bestdrag=model.predict(numpy.array([[0.0,0.0]]))[0][0]

log=[[0 for a in range(int(1/step+1))] for b in range(int(1/step+1))]

for a in range(int(1/step+1)):
    for b in range(int(1/step+1)):
        prediction=model.predict(numpy.array([[step*a,step*b]]))[0][0]
        log[a][b]=prediction
        if prediction<bestdrag:
            besta=step*a
            bestb=step*b
            bestdrag=prediction
        
        print(format("%d of %d complete" % (a/step+b+2,(1/step+1)*(1/step+1))))

print("Best a:", besta)
print("Best b:", bestb)
print("Best predicted drag:", bestdrag)

open(filepath+"/log.txt","w").write(format("Predicted %f,%f as the best with drag %f\n Actual drag: " % (besta,bestb,bestdrag)))

open(filepath+"/dataset.csv","a").write(format("\n%f,%f" % (besta,bestb)))

plt.imshow(log)
plt.colorbar()
plt.show()
plt.waitforbuttonpress(0)