import tensorflow as tf
import keras
import pathlib

outFilepath = str(pathlib.Path(__file__).resolve().parent)

model = keras.models.Sequential([
    keras.Input(shape=(1,)),
    keras.layers.Dense(128,activation=tf.nn.leaky_relu),
    keras.layers.Dense(128,activation=tf.nn.leaky_relu),
    keras.layers.Dense(1,activation="linear")
])
    
model.compile(optimizer='adam', loss='mean_squared_error')

model.save(outFilepath+"/model0.keras")

try:
    open(outFilepath+"/Iter0.tsv", "x").write("l\tdrag\tSA\tAltitude\n")
except FileExistsError:
    print("Dataset file already exists")