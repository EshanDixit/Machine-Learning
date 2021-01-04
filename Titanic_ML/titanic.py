import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import csv

#let's set male as 1 and female as 0
def updateGender(pandas_object):
    length_of_pandas_object = len(pandas_object)
    for i in range(length_of_pandas_object):
        if pandas_object["Sex"][i] == "male":
            pandas_object["Sex"][i] = 1
        else:
            pandas_object["Sex"][i] = 0
            
training_csv_file = pd.read_csv("train.csv")
testing_csv_file = pd.read_csv("test.csv")

X_train = training_csv_file.drop(columns = ["Survived", "Name", "PassengerId", "Ticket", "Cabin", "Embarked"]) #every columns except survived, name, 
updateGender(X_train) #converting string gender to float

Y_train = training_csv_file["Survived"] #output set - survived or not

X_test = testing_csv_file.drop(columns = ["Name", "Ticket", "Cabin", "Embarked"])
updateGender(X_test) #converting string gender to float
X_test = X_test.drop(X_test.columns[0], axis = 1) #dropping the unnecessary column

X_train = X_train.reset_index()
X_test  = X_test.reset_index()

model = DecisionTreeClassifier() #instantiaiting the model
model.fit(X_train, Y_train) #training the model

predictions = model.predict(X_test)

first_column_array = ["PassengerId"]
for i in range(892, 1310):
    first_column_array.append(str(i))


second_column_array = ["Survived"]
for i in predictions:
    second_column_array.append(str(i))


small_data_list = []
final_data_list = []
for i in range(419):
    small_data_list.append(first_column_array[i])
    small_data_list.append(second_column_array[i])
    final_data_list.append(small_data_list)
    small_data_list = []

with open("predictions.csv", "w", newline = "") as f:
    writer = csv.writer(f, delimiter = ",")
    writer.writerows(final_data_list)
