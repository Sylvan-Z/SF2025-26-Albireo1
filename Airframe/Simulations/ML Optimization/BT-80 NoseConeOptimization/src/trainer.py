import keras
import numpy as np
import shutil

inFilepath = input("Filepath In: ")
outFilepath = input("Filepath Out: ")

controlDrag = None
with open(inFilepath+"/dataset.csv", "r") as inputFile:
    controlDrag = float(inputFile.readlines()[1].strip().split(",")[-1])
    inputFile.close()

xSet=[]
ySet=[]
with open(inFilepath+"/dataset.csv", "r") as inputFile:
    inputList = inputFile.readlines()
    inputList.pop(0)
    for line in inputList:
        data = line.strip().split(",")
        xSet.append(list(map(float, data[:-1])))
        ySet.append(float(data[-1])-controlDrag)
    inputFile.close()
print(ySet)

shutil.copyfile(inFilepath+"/dataset.csv", outFilepath+"/dataset.csv")

model = keras.models.load_model(inFilepath+"/model.keras")

print(model)

model.fit(np.array(xSet,dtype=float), np.array(ySet,dtype=float), epochs=1000, shuffle=True, callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=200)])

model.evaluate(np.array(xSet,dtype=float), np.array(ySet,dtype=float))

model.save(outFilepath+"/model.keras")