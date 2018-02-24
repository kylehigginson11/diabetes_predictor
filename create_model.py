from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np

EPOCHS = 200

# apply random seed for consistency
np.random.seed(7)

# load data set
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# split into input and output
x_train = dataset[:,0:8]
y_train = dataset[:,8]

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit model
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=10)

# evaluate model
scores = model.evaluate(x_train, y_train)
print(str(model.metrics_names[1]) + " " + "%.2f" % round(scores[1]*100,2) + "%")

# save model
print ('Saving Model!')
model.save('diabetes_model.h5')
