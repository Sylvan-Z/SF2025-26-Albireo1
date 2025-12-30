import keras
import numpy as np
import shutil

inFilepath = input("Filepath In: ")
outFilepath = input("Filepath Out: ")

xSet=[]
ySet=[]
with open(inFilepath+"/dataset.csv", "r") as inputFile:
    inputList = inputFile.readlines()
    inputList.pop(0)
    for line in inputList:
        data = line.strip().split(",")
        xSet.append([float(data[0]), float(data[1])])
        ySet.append(float(data[2]))
    inputFile.close()

shutil.copyfile(inFilepath+"/dataset.csv", outFilepath+"/dataset.csv")

model = keras.models.load_model(inFilepath+"/model.keras")

print(model)

model.fit(np.array(xSet,dtype=float), np.array(ySet,dtype=float), epochs=5000, shuffle=True, batch_size=25, callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=100)])

model.evaluate(np.array(xSet,dtype=float), np.array(ySet,dtype=float))

model.save(outFilepath+"/model.keras")