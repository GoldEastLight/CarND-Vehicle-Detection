from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import numpy as np
import glob
import pickle
import os.path
import time
import cv2
from extract import FeatureExtractor
from tracker import Filter
from utils import draw_boxes, colors

train_parameters = {
    'c_color_space': 'YUV',
    'h_color_space': 'YUV',
    'orient': 10,
    'pix_per_cell': 8,
    'cell_per_block': 4,
    'hog_channel': 6, #3 0+1 # 4 0+2 # 5 1 + 2 #6 1+2+3
    'spatial_size': (32, 32),
    'hist_bins': 32,
    'spatial_feat': True,
    'hist_feat': True,
    'hog_feat': True,
}

model = {
    'svc': None,
    'X_scaler': None,
    'accuracy': None,
}

def train():
    extractor = FeatureExtractor(train_parameters, model)
    vehicles_files = glob.glob('./vehicles/**/*.png', recursive=True)
    non_vehicles_files = glob.glob('./non-vehicles/non-vehicles/Extras/*.png', recursive=True)
    # non_vehicles_files = glob.glob('./non-vehicles/**/*.png', recursive=True)

    n_cars = len(vehicles_files)
    n_notcars = len(non_vehicles_files)
    print('cars:', n_cars)
    print('notcars:', n_notcars)
    sample_size = min(n_cars, n_notcars)
    print('sample_size:', sample_size)
    cars = vehicles_files
    notcars = non_vehicles_files
    t = time.time()
    car_features = extractor.extract_features(cars)
    notcar_features = extractor.extract_features(notcars)

    X = np.vstack((car_features, notcar_features)).astype(np.float32)
    print(X.shape)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    print(np.max(scaled_X), np.min(scaled_X), np.max(X), np.min(X))
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Feature vector length:', len(X_train[0]))
    print('Training Set Size:', len(X_train))
    print('Test Set Size:', len(X_test))

    # parameters = {'kernel': ('linear', 'rbf'), 'C': [0.1, 1]}
    # svr = SVC()
    # clf = GridSearchCV(svr, parameters)
    # clf.fit(X_train, y_train)
    # Use a linear SVC
    t2 = time.time()
    svc = LinearSVC()
    # Check the training time for the SVC
    svc.fit(X_train, y_train)
    t3 = time.time()
    print(round(t3 - t2, 2), 'Seconds to train SVC...')
    print(round(t3 - t, 2), 'Seconds to total...')
    # Check the score of the SVC
    accuracy = round(svc.score(X_test, y_test), 4)
    print('Test Accuracy of SVC = ', accuracy)

    return svc, X_scaler, accuracy


def load_data():
    pickle_file = 'model.p'
    global model
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            param = pickle.load(f)
            model.update(param)
            print("Test Accuracy of SVC = ", model['accuracy'])
    else:
        svc, X_scaler, accuracy = train()
        model['svc'] = svc
        model['X_scaler'] = X_scaler
        model['accuracy'] = accuracy
        with open(pickle_file, 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

