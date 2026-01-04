import tensorflow as tf
import keras

outFilepath = input("Filepath Out: ")

model = keras.models.Sequential([
    keras.Input(shape=(3,)),
    keras.layers.Dense(128,activation=tf.nn.leaky_relu),
    keras.layers.Dense(64,activation=tf.nn.leaky_relu),
    keras.layers.Dense(32,activation=tf.nn.leaky_relu),
    keras.layers.Dense(1,activation="linear")
])
    
model.compile(optimizer='adam', loss='mean_squared_error')

model.save(outFilepath+"/model.keras")

try:
    open(outFilepath+"/dataset.csv", "x").write("a,b,c,drag\n")
except FileExistsError:
    print("Dataset file already exists")