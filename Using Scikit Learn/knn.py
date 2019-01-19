import pickle
import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

def unpickle(file):
    # unpickles cifar10 dataset
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic

def getCIFAR10(direc, filename, batches):
    # Converts the data in batches to a full training set
    for j in range(1, batches+1):
        file = direc + filename + str(j)
        dic = unpickle(file)
        if j == 1:
            X_train = dic[b'data']
            y_train = dic[b'labels']
        else:
            temp_X = dic[b'data']
            temp_y = dic[b'labels']
            X_train = np.concatenate((X_train, temp_X))
            y_train = np.concatenate((y_train, temp_y))
    return X_train, y_train

if __name__ == '__main__':
    direc = '../data/'
    test_file = 'test_batch'
    filename = 'data_batch_'
    X_train, y_train = getCIFAR10(direc, filename, 5)
    data_test = unpickle(direc + test_file)
    X_test = data_test[b'data']
    y_test = data_test[b'labels']
    # Call kNN
    k = int(input("Enter the value of k for k-Nearest Neighbor Classifier: "))
    print("Computation under process")
    print("Please Wait...")
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    Yte_predict = knn.predict(X_test)
    print("Prediction complete")
    print('The accuracy of classifier on test data: {:.2f}' .format((knn.score(X_test, y_test)*100)))
