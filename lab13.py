# Your existing code to prepare the data
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

train.fillna(train.select_dtypes(include=[np.number]).mean(), inplace=True)
test.fillna(test.select_dtypes(include=[np.number]).mean(), inplace=True)

train = train.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)
test = test.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)

labelEncoder = LabelEncoder()
labelEncoder.fit(train['Sex'])
train['Sex'] = labelEncoder.transform(train['Sex'])
labelEncoder.fit(test['Sex'])
test['Sex'] = labelEncoder.transform(test['Sex'])

X = np.array(train.drop(['Survived'], axis=1).astype(float))
y = np.array(train['Survived'])

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2, max_iter=600, algorithm = 'lloyd', n_init=10, random_state=42)
kmeans.fit(X_scaled)

# We need to count for both possible label mappings
correct_count_1 = 0
correct_count_2 = 0

for i in range(len(X_scaled)):
    # Use the scaled data point for prediction
    predict_me = np.array(X_scaled[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)

    # Check against the ground truth 'y[i]'
    if prediction[0] == y[i]:
        correct_count_1 += 1
    else:
        # If it doesn't match, it must match the "flipped" mapping
        correct_count_2 += 1

# The correct accuracy is the maximum of the two possibilities
accuracy = max(correct_count_1, correct_count_2) / len(X_scaled)

print(accuracy)