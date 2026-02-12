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

bestFeatures=[0.5,0.75]
bestdrag=model.predict(np.array([bestFeatures]))[0][0]+controlDrag

x=[bestFeatures[0]]
y=[bestFeatures[1]]
z=[bestdrag]


deltaF=0.01
deltaT=70
residuals=[float("inf")]
iterations=0
while sum(map(lambda x: abs(x), residuals))/len(residuals)!=0.000000 and iterations<300:
    allDFeatures=[]
    for i in range(len(bestFeatures)):
        allDFeatures.append(bestFeatures.copy())
        allDFeatures[-1][i]+=deltaF
    derivitives=list(map(lambda x:bestdrag-x[0]-controlDrag, model.predict(np.array(allDFeatures))))
    newFeatures=[]
    residuals=[]
    for i in range(len(bestFeatures)):
        newFeatures.append(bestFeatures[i]+derivitives[i]*deltaT)
        newFeatures[-1]=np.clip(newFeatures[-1],0,1)
        residuals.append(newFeatures[-1]/deltaT-bestFeatures[i]/deltaT)
    bestFeatures=newFeatures.copy()
    bestdrag=model.predict(np.array([bestFeatures]))[0][0]+controlDrag
    iterations+=1
    print("Iteration %d: Predicted %s as the best with drag %f, average residual %f" % (iterations, str(bestFeatures), bestdrag, sum(map(lambda x: abs(x), residuals))/len(residuals)))
    x.append(bestFeatures[0])
    y.append(bestFeatures[1])
    z.append(bestdrag)

fig.add_trace(go.Scatter3d(x=list(x), y=list(y), z=list(z), mode='markers+lines', marker=dict(size=2, color='black')))

fig.update_layout(
    scene = dict(
        xaxis = dict(nticks=4, range=[0,1],),
        yaxis = dict(nticks=4, range=[0,1],),
        zaxis = dict(nticks=4, range=[-205,-202],),
        aspectmode='cube'))

fig.show()