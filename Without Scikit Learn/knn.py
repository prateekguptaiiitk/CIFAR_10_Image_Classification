import numpy as np
import pickle

class KNearestNeighbor(object):
    
    def __init__(self):
        pass

    def train(self, X, y):
       
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
       
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)
        
        return self.predict_labels(dists, k=k)
    
    def compute_distances_two_loops(self, X):
        
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                
                dists[i, j] = np.sqrt(np.sum((X[i] - self.X_train[j])**2))

        return dists
            
    def compute_distances_one_loop(self, X):
        
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):

            dists[i] = np.sqrt(np.sum((X[i] - self.X_train)**2, axis=1))
               
        return dists
    def compute_distances_no_loops(self, X):
        
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        
        dists = np.sqrt((X**2).sum(axis=1, keepdims=True) + (self.X_train**2).sum(axis=1) - 2 * X.dot(self.X_train.T))
        
        return dists

    def predict_labels(self, dists, k=1):
        
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
           
            closest_y = []
           
            closest_y = self.y_train[np.argsort(dists[i])][:k]
            
            y_pred[i] = np.argmax(np.bincount(closest_y))
                 
        return y_pred

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
    
    # Call NN
    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)
    
    # Choice of computation
    dists = classifier.compute_distances_two_loops(X_test)
    k = int(input("Enter the value of k for k Nearest Neighbor Classifier: "))
    y_test_pred = classifier.predict_labels(dists, k)
    
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / 1000
    print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

