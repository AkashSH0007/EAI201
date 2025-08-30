#import tensorflow and numpy
#sample data
#define a model
#compile with optimizer and loss function
#predicted

import tensorflow as tf
import numpy as np

x=np.array([1,2,3,4,5], dtype=float)
y=np.array([3,5,7,9,11],dtype=float)

model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd',loss='mean_squared_error')

model.fit(x,y,epochs=100,verbose=0)

predicted=model.predict([10.0])
print(f'prediction for x=10:{predicted[0][0]}')
