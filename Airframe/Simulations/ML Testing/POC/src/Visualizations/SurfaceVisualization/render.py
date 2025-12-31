import keras
import numpy as np
import itertools
import plotly.graph_objects as go

filepath = input("Filepath: ")

model = keras.models.load_model(filepath+"/model.keras")

controlDrag = None
with open(filepath+"/dataset.csv", "r") as inputFile:
    controlDrag = float(inputFile.readlines()[1].strip().split(",")[2])
    inputFile.close()

step=0.01
allFeatures = list(map(list, itertools.product(list(np.arange(0, 1+step, step)), repeat=2)))
allDrags = list(map(lambda x: x[0]+controlDrag, model.predict(np.array(allFeatures))))

x=map(lambda x: x[0], allFeatures)
y=map(lambda x: x[1], allFeatures)
z=allDrags

fig = go.Figure()

fig.add_trace(go.Mesh3d(x=list(x),
                   y=list(y),
                   z=list(z),
                   opacity=0.5,
                   colorscale='RdBu',
                   intensity=list(z),
                   showscale=True))

x=[]
y=[]
z=[]
with open(filepath+"/dataset.csv", "r") as inputFile:
    inputList = inputFile.readlines()
    inputList.pop(0)
    for line in inputList:
        data = line.strip().split(",")
        x.append(float(data[0]))
        y.append(float(data[1]))
        z.append(float(data[2]))
    inputFile.close()

fig.add_trace(go.Scatter3d(x=list(x), y=list(y), z=list(z), mode='markers', marker=dict(size=2, color='black')))

fig.update_layout(
    scene = dict(
        xaxis = dict(nticks=4, range=[0,1],),
        yaxis = dict(nticks=4, range=[0,1],),
        zaxis = dict(nticks=4, range=[-205,-202],),
        aspectmode='cube'))

fig.show()