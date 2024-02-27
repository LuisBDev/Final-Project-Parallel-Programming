import csv
from collections import Counter
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import os


class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        valid = self.k <= len(X)
        if not valid:
            raise Exception('El valor de k debe ser menor que el tamaño de los datos de entrenamiento.')

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return predicted_labels

    def _predict(self, x):
        # compute distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # get k nearest samples, labels
        k_indices = np.argsort(distances)[0:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # majority vote, most common label
        most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common_label

    def getLocalSortedDistances(self, X):
        k_local_sorted_distances = [self._getLocalSortedDistances(x) for x in X]
        return k_local_sorted_distances

    def _getLocalSortedDistances(self, x):
        # array of tuples (distance, label)
        distances = []
        for i in range(0, len(self.X_train)):
            distance = (euclidean_distance(x, self.X_train[i]), self.y_train[i])
            distances.append(distance)
        # sort the array in ascending order by distance and pick k first elements
        k_local_sorted_distances = sorted(distances, key=lambda t: t[0])[0:self.k]
        return k_local_sorted_distances

def euclidean_distance(p, q):
    return np.sqrt(np.sum((p-q)**2))

def load_data():
    
    training_path = os.getcwd() + '/data/credit_train.csv'
    test_path = os.getcwd() + '/data/credit_test.csv'
    training_data = pd.read_csv(training_path)
    test_data = pd.read_csv(test_path)
    return training_data, test_data

def extract_data(data_records, training_data, test_data):
    # Extract a number of records according to testing cases
    X_train = training_data.iloc[:data_records,:5]
    y_train = training_data.iloc[:data_records,(training_data.shape[1]-1)]
    X_test = test_data.iloc[:500,:5]
    y_test = test_data.iloc[:500,(test_data.shape[1]-1)]  
    # transform into numpy array for faster computation
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    return X_train, y_train, X_test, y_test


def run_parallel_KNN(n_jobs, k, X_train, y_train, X_test, y_test):
    knn = KNN(k)
    knn.fit(X_train, y_train)

    def calculate_local_distances(x):
        return knn.predict(np.array([x]))[0]  

    local_predictions = Parallel(n_jobs=n_jobs)(delayed(calculate_local_distances)(x) for x in X_test)
    
    accuracy = accuracy_score(y_test, local_predictions)
    
    return accuracy, local_predictions

def run_sequential_KNN(k, X_train, y_train, X_test, y_test):
    knn = KNN(k)
    knn.fit(X_train, y_train)
    local_predictions = knn.predict(X_test) 
    accuracy = accuracy_score(y_test, local_predictions)
    return accuracy, local_predictions


csv_path = 'results.csv'  

cant_filas = 20000
n_jobs = 6
k = 5


training_data, test_data = load_data()
X_train, y_train, X_test, y_test = extract_data(cant_filas, training_data, test_data)


############################# Parallel Version ############################
start_time = time.time()
accuracy_parallel, parallel_predictions = run_parallel_KNN(n_jobs=n_jobs, k=k, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
end_time = time.time()
execution_time_parallel = end_time - start_time



############################# Sequential Version ############################
start_time = time.time()
accuracy_sequential, sequential_predictions = run_sequential_KNN(k, X_train, y_train, X_test, y_test)
end_time = time.time()
execution_time_sequential = end_time - start_time



print("\n\n-------------------------------------------------------------------")
print(f"Training the model with {cant_filas} records and testing with {len(X_test)} records")
print(f"Number of predictions: {len(parallel_predictions)}")
print(f"Number of jobs: {n_jobs}")


# Calculate total execution time
print(f"Total execution time for parallel version: {execution_time_parallel}")
print(f"Total execution time for sequential version: {execution_time_sequential}")

# Calculate speedup
speedup = execution_time_sequential / execution_time_parallel
print(f"Speedup: {speedup}")

# Calcular la eficiencia
eficiencia = speedup / n_jobs

print(f"Efficiency: {eficiencia}")

#Overall speedup using amdahls law
p = 0.2
overall_speedup = 1 / ((1-p) + p/n_jobs)
print(f"Overall speedup: {overall_speedup}")
print("-------------------------------------------------------------------\n")

# List with results
new_row = [n_jobs, cant_filas, execution_time_sequential, execution_time_parallel, speedup, eficiencia]

# Añadir resultados al archivo CSV existente sin comillas
with open(csv_path, 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(new_row)
