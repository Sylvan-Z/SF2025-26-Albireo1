import numpy as np
import shutil
from typing import Callable
import keras
import itertools
import pandas as pd

class Iteration:
    def __init__(self, filepath:str, n:int, batchsize:int=25):
        self.n=int(n)
        self.parentFilepath=filepath
        self.reloadData()
        self.reloadModel()
        self.updateControl()
        self.batchsize=batchsize
    

    #Data Handling
    def generateDeltas(self):
        self.updateControl()
        self.deltas=pd.DataFrame(columns=self.df.columns)
        for column in self.deltas.columns.to_list():
            self.deltas[column]=self.df[column]-self.control[column]
    
    def reloadData(self):
        self.df=pd.read_csv(self.parentFilepath+f"/iter{self.n}.tsv",sep='\t')

    def updateControl(self):
        self.control=self.df.iloc[0] 

    def addRow(self,row:dict[str,float]):
        self.df.loc[len(self.df)]=row

    def saveCsv(self):
        self.df.to_csv(path_or_buf=self.parentFilepath+f"/iter{self.n}.tsv", index=False, sep='\t')


    #Model handling
    def reloadModel(self):
        self.model:keras.Sequential=keras.models.load_model(self.parentFilepath+f"/model{self.n}.keras")

    def calculateNulls(self,key:str, func:Callable[[pd.Series,pd.Series],float]):
        self.updateControl()
        for row in range(len(self.df)):
            if self.df.at[row,key]==-1 or self.df.at[row,key]==None or pd.isna(self.df.at[row,key]):
                self.df.at[row,key]=func(self.df.iloc[row],self.control)

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

    def optimize(self,maximize:bool, *xSamples, deltaT=0.005):
        xSample=np.array(list(itertools.product(*xSamples)),dtype=float)
        print(xSample)
        ySample=self.model.predict(xSample)
        print(ySample)

        bestXs:list=xSample[0]
        bestY:list=ySample[0]

        for i in range(len(ySample)):
            if (bestY<ySample[i])==maximize:
                bestXs=xSample[i].tolist()
                bestY=ySample[i][0]
        
        print("Done broad search")
        print(format("Predicted %s as the best X with Y %f" % (str(bestXs), bestY)))
        print("Starting gradient descent refinement")

        deltaF=0.01
        if(maximize):deltaT=-deltaT
        residuals=[float("inf")]
        iterations=0

        while sum(map(lambda x: abs(x), residuals))/len(residuals)>0.0001:
            try:
                print(f"Best Xs {bestXs}")
                dXs=[]
                for i in range(len(bestXs)):
                    dXs.append(bestXs.copy())
                    dXs[-1][i]+=deltaF
                print(f"dXs {dXs}")
                dYs=list(map(lambda x:(bestY-x[0])/deltaF, self.model.predict(np.array(dXs))))
                print(f"dYs {dYs}")
                newFeatures=[]
                residuals=[]
                for i in range(len(bestXs)):
                    print("bestXs[i]",str(bestXs[i]))
                    print("(dYs[i]*deltaT)",str((dYs[i]*deltaT)))
                    print("deltaT",str(deltaT))
                    newFeatures.append(bestXs[i]+(dYs[i]*deltaT))
                    print(newFeatures)
                    newFeatures[-1]=np.clip(newFeatures[-1],min(xSamples[i]),max(xSamples[i]))
                    residuals.append(newFeatures[-1]-bestXs[i])
                bestXs=newFeatures.copy()
                bestY=self.model.predict(np.array([bestXs],dtype=float))[0][0]
                iterations+=1
                print("Iteration %d: Predicted %s as the best X with Y %f, average residual %f" % (iterations, str(list(bestXs)), bestY, sum(map(lambda x: abs(x), residuals))/len(residuals)))
            except KeyboardInterrupt:
                break

        print(f"Optimization completed, bestXs: {bestXs}")
        return bestXs
        
    def saveModel(self):
        self.model.save(self.parentFilepath+f"/model{self.n}.keras")

    def saveAll(self):
        self.saveCsv()
        self.saveModel()
    
    def nextIteration(self):
        shutil.copy(self.parentFilepath+f"/model{self.n}.keras",self.parentFilepath+f"/model{self.n+1}.keras")
        shutil.copy(self.parentFilepath+f"/Iter{self.n}.tsv",self.parentFilepath+f"/Iter{self.n+1}.tsv")
        iteration=Iteration(self.parentFilepath,self.n+1)
        return iteration