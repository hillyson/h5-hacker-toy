# Data Structures
class Node:
    def __init__(self, data=None):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.length = 0
    
    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node
        self.length += 1
    
    def prepend(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.next = self.head
            self.head = new_node
        self.length += 1
    
    def delete(self, data):
        if not self.head:
            return
        
        if self.head.data == data:
            self.head = self.head.next
            self.length -= 1
            return
        
        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                self.length -= 1
                return
            current = current.next
    
    def search(self, data):
        current = self.head
        while current:
            if current.data == data:
                return current
            current = current.next
        return None
    
    def __str__(self):
        values = []
        current = self.head
        while current:
            values.append(str(current.data))
            current = current.next
        return ' -> '.join(values)

# Machine Learning Example
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Data Processing Pipeline
class DataPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
    
    def fit_transform(self, data):
        scaled = self.scaler.fit_transform(data)
        return self.pca.fit_transform(scaled)
    
    def transform(self, data):
        scaled = self.scaler.transform(data)
        return self.pca.transform(scaled)

# Performance Optimization
@jit(nopython=True)
def numba_optimized_sum(arr):
    total = 0.0
    for i in range(len(arr)):
        total += arr[i]
    return total

# Algorithms
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def binary_search(arr, target):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

# File Operations
def read_file(filename):
    try:
        with open(filename, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"File {filename} not found")
        return None

def write_file(filename, content):
    with open(filename, 'w') as file:
        file.write(content)

# Web Scraping
import requests
from bs4 import BeautifulSoup

def scrape_website(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return None

# Database Operations
import sqlite3

def create_database(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users
                      (id INTEGER PRIMARY KEY, name TEXT, email TEXT)''')
    conn.commit()
    conn.close()

def insert_user(db_name, name, email):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)", (name, email))
    conn.commit()
    conn.close()

# Machine Learning
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix

def train_model(X, y, model_type='random_forest'):
    if model_type == 'random_forest':
        model = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=100, random_state=42)
        )
    elif model_type == 'svm':
        model = make_pipeline(
            StandardScaler(),
            SVC(kernel='rbf', probability=True)
        )
    elif model_type == 'gradient_boosting':
        model = make_pipeline(
            StandardScaler(),
            GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
        )
    elif model_type == 'neural_network':
        model = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
        )
    
    model.fit(X, y)
    return model

def evaluate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5)
    y_pred = model.predict(X)
    
    return {
        'mean_accuracy': np.mean(scores),
        'std_accuracy': np.std(scores),
        'classification_report': classification_report(y, y_pred),
        'confusion_matrix': confusion_matrix(y, y_pred)
    }

def hyperparameter_tuning(X, y, model_type='random_forest'):
    if model_type == 'random_forest':
        param_grid = {
            'randomforestclassifier__n_estimators': [50, 100, 200],
            'randomforestclassifier__max_depth': [None, 10, 20]
        }
    elif model_type == 'svm':
        param_grid = {
            'svc__C': [0.1, 1, 10],
            'svc__gamma': [0.1, 1, 'scale']
        }
    
    model = train_model(X, y, model_type)
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X, y)
    return grid_search.best_params_

# Data Analysis
import pandas as pd

def analyze_data(csv_file):
    df = pd.read_csv(csv_file)
    return {
        'mean': df.mean(),
        'median': df.median(),
        'std': df.std()
    }

# Web Development
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the API"

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify({"data": "Sample response"})

# Testing
import unittest

class TestLinkedList(unittest.TestCase):
    def setUp(self):
        self.ll = LinkedList()
    
    def test_append(self):
        self.ll.append(1)
        self.assertEqual(self.ll.head.data, 1)
        self.assertEqual(self.ll.tail.data, 1)
    
    def test_prepend(self):
        self.ll.prepend(1)
        self.assertEqual(self.ll.head.data, 1)
        self.assertEqual(self.ll.tail.data, 1)

# Utility Functions
def timer(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} executed in {end-start:.4f} seconds")
        return result
    return wrapper

def memoize(func):
    cache = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

# Main Execution
if __name__ == "__main__":
    # Data Structures Example
    ll = LinkedList()
    ll.append(1)
    ll.append(2)
    ll.prepend(0)
    print(f"Linked List: {ll}")
    
    # Algorithms Example
    arr = [5, 3, 8, 4, 2]
    sorted_arr = bubble_sort(arr.copy())
    print(f"Sorted Array: {sorted_arr}")
    print(f"Binary Search for 4: {binary_search(sorted_arr, 4)}")
    
    # Machine Learning Example
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    model = train_model(X, y)
    metrics = evaluate_model(model, X, y)
    print(f"Model Accuracy: {metrics['mean_accuracy']:.2f} ± {metrics['std_accuracy']:.2f}")