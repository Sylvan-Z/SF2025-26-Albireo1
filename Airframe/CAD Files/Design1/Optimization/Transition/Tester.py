import pathlib
from src.OptimizationToolkit import *
from src.RocketSimToolkit import *

iteration=Iteration(str(pathlib.Path(__file__).resolve().parent), 1)

iteration.updateControl()

print(iteration.model.predict(np.array([[1,0,0.5]])))

print(iteration.model.predict(np.array([[1,0,0.5]]))[0]*50+iteration.control["Altitude"])