import numpy as np
import shutil
import csv
from typing import Callable
import keras

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
                print("Training completed")

    def optimize(self,maximize:bool,*args):
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
