import numpy as np

train_path = 'C:/Users/ldmag/Downloads/iris-training-data.csv'
test_path = 'C:/Users/ldmag/Downloads/iris-testing-data.csv'

# Arrays for train / test labels
training_labels = np.loadtxt(train_path, dtype = 'str', delimiter = ',', usecols = (4))
test_labels = np.loadtxt(test_path, dtype = 'str', delimiter = ',', usecols = (4))

# Arrays for train / test data
training_data = np.loadtxt(train_path, dtype = 'f8', delimiter = ',', usecols = (0,1,2,3))
test_data = np.loadtxt(test_path, dtype = 'f8', delimiter = ',', usecols = (0,1,2,3))

X_train, X_test, Y_train, Y_test = training_data, test_data, training_labels, test_labels

# K-NN function with Eucl distance
def knn(X_train, X_test, k):
    distances = -2 * X_train@X_test.T + np.sum(X_test**2, axis =1) + np.sum(X_train**2, axis =1)[:, np.newaxis]
    distances[distances < 0] = 0
    distances = distances**.5
    # distances = np.linalg.norm(X_train - X_test)
    indices = np.argsort(distances, 0)
    distances = np.sort(distances,0)

    return indices[0:k,:], distances[0:k,:]

# Predictions
def knn_predictions(X_train,Y_train,X_test,k=3):
    indices, distances = knn(X_train,X_test,k)
    Y_train = Y_train.flatten()
    rows, columns = indices.shape
    predictions = list()
    for j in range(columns):
        temp = list()
        for i in range(rows):
            cell = indices[i][j]
            temp.append(Y_train[cell])
        predictions.append(max(temp, key=temp.count))
    predictions = np.array(predictions)
    return predictions

predictions = knn_predictions(X_train, Y_train, X_test, 3)

# Accuracy function , 5 fold
def accuracy(Y_test, predictions):
    x = Y_test.flatten() == predictions.flatten()
    grade = np.mean(x)
    return np.round(grade*100,2)

# Result
print('DATA-51100', 'Spring 2020')
print('Lionel Dsilva')
print('Programming Assignment 3\n')

print('# ', 'True ', 'Predictions')

for x,v,z in zip(range(0,76),Y_test, predictions):
    print(x,v,z)

print('Accuracy: ', accuracy(knn_predictions(X_train, Y_train, X_test, 5), Y_test), '%')
