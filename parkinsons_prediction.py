import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'MDVP:Fo(Hz)': [119.992, 122.400, 116.682, 116.676, 116.014],
    'MDVP:Fhi(Hz)': [157.302, 148.650, 131.111, 137.871, 141.781],
    'MDVP:Flo(Hz)': [74.997, 113.819, 111.555, 111.366, 110.655],
    'status': [1, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

X = df.drop('status', axis=1)
Y = df['status']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

model = LogisticRegression()

model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print("Accuracy Score:", training_data_accuracy)

input_data = (120.0, 150.0, 75.0)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)

if prediction[0] == 1:
    print("The person has Parkinson's Disease")
else:
    print("The person does not have Parkinson's Disease")
