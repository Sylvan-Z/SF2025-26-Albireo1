import numpy as np
import shutil
from typing import Callable
import keras
import itertools
import pandas as pd

class Iteration:
    def __init__(self, filepath:str, n:int):
        self.n=int(n)
        self.parentFilepath=filepath
        self.reloadData()
        self.updateControl()

    def generateDeltas(self):
        self.updateControl()
        self.deltas=pd.DataFrame(columns=self.df.columns)
        for column in self.deltas.columns.to_list():
            self.deltas[column]=self.df[column]-self.control[column]
    
    def reloadData(self):
        self.df=pd.read_csv(self.parentFilepath+f"/iter{self.n}.csv")

    def updateControl(self):
        self.control=self.df.iloc[0] 

    def calculateNulls(self,key:str, func:Callable[[pd.Series,pd.Series],float]):
        for row in range(len(self.df)):
            if self.df.at[row,key]==-1:
                self.df.at[row,key]=func(self.df.iloc[row],self.control)
        
    def writeAll(self):
        self.df.to_csv(path_or_buf=self.parentFilepath+f"/iter{self.n}.csv")

    def nextIteration(self):
        shutil.copy(self.parentFilepath+f"/Iter{self.n}.csv",self.parentFilepath+f"/Iter{self.n+1}.csv")
        iteration=Iteration(self.parentFilepath,self.n+1)
        return iteration
    
class Model:
    def __init__(self,filepath,batchsize:int=25):
        self.filepath=filepath
        self.batchsize=batchsize
        self.reloadModel()
    
    def reloadModel(self):
        self.model:keras.Sequential=keras.models.load_model(self.filepath+"/model.keras")

    def fit(self,xsets:list[list[float]],yset:list[float]):
        xsets=np.array(xsets, dtype=float)
        yset=np.array(yset, dtype=float)
        prevModel=self.model
        print(xsets)
        while True:
            print(self.model.evaluate(xsets,yset))
            epochs=int(input("Epochs (>0:epochs, =0:Break, =-1:Go back): "))
            if epochs==0:
                break
            elif epochs<0:
                self.model=prevModel
                print("Training reverted")
            else:
                self.model.fit(xsets,yset,self.batchsize,epochs)
                prevModel=self.model
                print("Training completed")

    def optimize(self,maximize:bool,*xSamples):
        xSample=np.array(list(itertools.product(*xSamples)),dtype=float)
        print(xSample)
        ySample=self.model.predict(xSample,batch_size=self.batchsize)
        bestXs=xSample[0]
        bestY=ySample[0]
        for i in range(len(ySample)):
            if bestY>ySample[i]==maximize:
                bestXs=xSample[i]
                bestY=ySample[i]
        
        print("Done broad search")
        print(format("Predicted %s as the best X with Y %f" % (str(bestXs), bestY)))
        print("Starting gradient descent refinement")

        deltaF=0.001
        if(maximize):deltaT=-0.0005 
        else: deltaT=0.0005
        residuals=[float("inf")]
        iterations=0

        while sum(map(lambda x: abs(x), residuals))/len(residuals)>0.00003:
            bestDXs=[]
            for i in range(len(bestXs)):
                bestDXs.append(bestXs.copy())
                bestDXs[-1][i]+=deltaF
            derivitives=list(map(lambda x:(bestY-x[0])/deltaF, self.model.predict(np.array(bestDXs))))
            newFeatures=[]
            residuals=[]
            for i in range(len(bestXs)):
                newFeatures.append(bestXs[i]+derivitives[i]*deltaT)
                newFeatures[-1]=np.clip(newFeatures[-1],min(xSample[i]),max(xSample[i]))
                residuals.append(newFeatures[-1]-bestXs[i])
            bestXs=newFeatures.copy()
            bestdrag=self.model.predict(np.array(bestXs,dtype=float))[0][0]
            iterations+=1
            print("Iteration %d: Predicted %s as the best X with Y %f, average residual %f" % (iterations, str(bestXs), bestdrag, sum(map(lambda x: abs(x), residuals))/len(residuals)))

            return bestXs
        
    def save(self):
        self.model.save(self.filepath+"/model.keras")