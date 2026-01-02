import tensorflow as tf
import keras

outFilepath = input("Filepath Out: ")

inputs=tf.keras.Input(shape=(2,))

model = keras.models.Sequential([
    keras.layers.Dense(128,input_shape=(2,)),
    keras.layers.Dense(128,activation=tf.nn.leaky_relu),
    keras.layers.Dense(128,activation=tf.nn.leaky_relu),
    keras.layers.Dense(128,activation=tf.nn.leaky_relu),
    keras.layers.Dense(1,activation="linear")
])
    
model.compile(optimizer='adam', loss='mean_squared_error')

model.save(outFilepath+"/model.keras")

try:
    open(outFilepath+"/dataset.csv", "x").write("a,b,drag\n")
except FileExistsError:
    print("Dataset file already exists")