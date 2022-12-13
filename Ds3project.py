#importing the dependencies

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from  sklearn.ensemble import RandomForestClassifier
import warnings
import pickle
warnings.filterwarnings("ignore")
# loading the dataset to a Pandas DataFrame
wine_dataset = pd.read_csv('winequality-resd.csv')

correlation = wine_dataset.corr()

# constructing a heatmap to understand the correlation between the columns
plt.figure(figsize=(10,10))

# separate the data and Label
X = wine_dataset.drop(['quality', 'alcohol','pH','density','citric acid'], axis=1)
Y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

model = RandomForestClassifier()

model.fit(X_train, Y_train)


pickle.dump(model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))














