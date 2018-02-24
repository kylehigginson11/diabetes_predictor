from io import StringIO

from keras.models import load_model
import numpy as np

data = ""
data += input('Number of times pregnant: ') + ","
data += input('Plasma glucose concentration: ') + ","
data += input('blood pressure in mm Hg: ') + ","
data += input('Tricep skinfold thickness: ') + ","
data += input('2-hour serum insulin: ') + ","
data += input('body mass index: ') + ","
data += input('diabetes pedigree function: ') + ",";
data += input('age: ') + " "

data = StringIO(data)

# load pima indians dataset
dataset = np.loadtxt(data, delimiter=",")
dataset = np.reshape(dataset, (-1, 8))

X = dataset[:,0:8]
print(X)

# returns a compiled model
# identical to the previous one
model = load_model('diabetes_model.h5')

# calculate predictions
prediction = model.predict(dataset)
# round predictions
rounded = round(prediction[0][0])

# Print precitions out in understandable terms
if rounded == 1:
    print("You are likely to develop diabetes")
else:
    print("It is unlikely that you will develop diabetes")
