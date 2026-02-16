import numpy as np
import shutil
import csv
from typing import Callable
import keras
import itertools

class Iteration:
    def __init__(self, filepath:str, n:int):
        self.n=int(n)
        self.parentFilepath=filepath
        self.csvFile=open(filepath+f"/Iter{n}.csv",'a+')
        self.csvFile.seek(0)
        self.reader=csv.DictReader(self.csvFile)
        self.writer=csv.DictWriter(self.csvFile,self.reader.fieldnames)
        self.readAll()
        self.control=self.rows[0]
        self.generateDeltas()

    def readAll(self):
        self.rows:list[dict[str,float]]=[]
        self.columns:dict[str,list[float]]=dict.fromkeys(self.reader.fieldnames,[])
        self.resetReader()
        for row in self.reader:
            self.rows.append({key:float(row[key]) for key in row.keys()})
            self.columns={key:self.columns[key]+[float(row[key])] for key in row.keys()}

    def generateDeltas(self):
        self.deltasRows=[{self.columns[key][i]-self.control[key] for key in self.reader.fieldnames} for i in range(len(self.rows))]
        self.deltasColumns:dict[str,list[float]]={key:[self.columns[key][i]-self.control[key] for i in range(len(self.columns[key]))] for key in self.reader.fieldnames}

    def resetReader(self):
        self.reader.line_num=0
    
    def scanForNulls(self,key:str):
        nulls=[]
        for i in range(len(self.rows)):
            if self.rows[i][key]==-1:
                nulls.append(i)
        return nulls
    
    def calculateNulls(self,key:str, func:Callable[[dict[str,float],dict[str,float]],float]):
        for i in self.scanForNulls(key):
            self.rows[i][key]=func(self.control,self.rows[i])
        self.writeAll()
        self.readAll()
        
    def writeAll(self, key:str):
        self.csvFile.seek(0)
        self.csvFile.truncate()
        self.csvFile.seek(0)
        self.writer.writeheader()
        self.writer.writerows(self.rows)

    def close(self):
        self.csvFile.close()

    def nextIteration(self):
        shutil.copy(self.parentFilepath+f"/Iter{self.n}.csv",self.parentFilepath+f"/Iter{self.n+1}.csv")
        iteration=Iteration(self.parentFilepath,self.n+1)
        self.close()
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
        xSample=np.array(itertools.product(*xSamples),dtype=float)
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
                newFeatures[-1]=np.clip(newFeatures[-1],0,max(xSample[i]))
                residuals.append(newFeatures[-1]-bestXs[i])
            bestXs=newFeatures.copy()
            bestdrag=self.model.predict(np.array([bestXs]))[0][0]
            iterations+=1
            print("Iteration %d: Predicted %s as the best X with Y %f, average residual %f" % (iterations, str(bestXs), bestdrag, sum(map(lambda x: abs(x), residuals))/len(residuals)))

            return bestXs