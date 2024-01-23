import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from KNN import KNN

# Load the Excel file
df = pd.read_excel('C:\\Users\\Asus\\Desktop\\KNN\\data.xlsx')

# Assuming the last column represents labels
X =  df.iloc[:, :2].values  # Features
y = df.iloc[:, -1].values    # Labels

# Visualize the dataset
# plt.figure(figsize=(8, 6))

# # Assuming X has two features for visualization
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
# plt.title('Visualization of the Dataset')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.show()


# Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


# Take input from the user
sepal_length = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))


# Create an array with the user input
user_input = np.array([[sepal_length, sepal_width]])





# k-NN model
clf = KNN(k=5)
# clf.fit(X_train, y_train)
clf.fit(X, y)
predictions = clf.predict(user_input)

print(predictions)

# Calculate accuracy
# acc = np.sum(predictions == y_test) / len(y_test)
# print(f'Accuracy: {acc}')
