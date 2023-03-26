import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load dataset into pandas dataframe
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
data = pd.read_csv(url, header=None)
data.head()
print(data)

# Check for duplicate values
duplicates = data.duplicated()
num_duplicates = sum(duplicates)

if num_duplicates > 0:
    # Print duplicate rows
    print(f"Found {num_duplicates} duplicates in the dataset:")
    print(data[duplicates])
else:
    print("No duplicates found in the dataset.")

# Drop duplicates
data = data.drop_duplicates()
print(data)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data.iloc[:,-1], test_size=0.3)

# Scale the features using RobustScaler
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# After Robustscaler
scaled_data = scaler.fit_transform(data)
data =pd.DataFrame(data=scaled_data, columns= data.columns)
print(data.head())

# Instantiate a DecisionTreeClassifier object with the desired hyperparameters
out = DecisionTreeClassifier(max_depth=5)

# Train the decision tree classifier using the training set
out.fit(X_train, y_train)

# Use the trained model to make predictions on the testing set
y_pred = out.predict(X_test)

# Evaluate the performance of the model using appropriate metrics

print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1-score:', f1_score(y_test, y_pred))
print('classification_report')
# Generate a classification report
cls_report = classification_report(y_test, y_pred)
print(cls_report)
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.title('Confusion matrix')
plt.xlabel('true class')
plt.ylabel('predicted class')
plt.show()

# Plot the decision tree
plt.figure(figsize=(10,10))
plot_tree(out, filled=True)
plt.show()
