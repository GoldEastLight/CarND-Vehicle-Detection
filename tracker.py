from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import numpy as np
import glob
import pickle
import os.path
import time
from extract import FeatureExtractor
from filter import Filter
from utils import draw_boxes

train_parameters = {
    'color_space': 'YUV',
    'orient': 10,
    'pix_per_cell': 8,
    'cell_per_block': 4,
    'hog_channel': 'ALL',
    'spatial_size': (32, 32),
    'hist_bins': 32,
    'cells_per_step': 2,
    'spatial_feat': True,
    'hist_feat': False,
    'hog_feat': True,
}

model = {
    'svc': None,
    'X_scaler': None,
}


def train():
    extractor = FeatureExtractor(train_parameters, model)
    vehicles_files = glob.glob('./vehicles/**/*.png', recursive=True)
    non_vehicles_files = glob.glob('./non-vehicles/non-vehicles/Extras/*.png', recursive=True)

    n_cars = len(vehicles_files)
    n_notcars = len(non_vehicles_files)
    print('cars:', n_cars)
    print('notcars:', n_notcars)
    sample_size = max(n_cars, n_notcars)
    # sample_size = -1
    cars = vehicles_files[0:sample_size]
    notcars = non_vehicles_files[0:sample_size]

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
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    return svc, X_scaler


def load_data():
    pickle_file = 'model.p'
    global model
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            param = pickle.load(f)
            model.update(param)
    else:
        svc, X_scaler = train()
        model['svc'] = svc
        model['X_scaler'] = X_scaler
        with open(pickle_file, 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


class Tracker():
    def __init__(self, parameters=train_parameters, model=model):
        self.extractor = FeatureExtractor(parameters, model)
        self.filter = Filter((720, 1280, 3))
        self.scale_configs = (
            (380, 500, 1, 1),
            (500, 650, 1.2, 2),
            # (600, 700, 1.5, 2),
            # (600, 700, 1.5)
        )

    def detect_car(self, image):
        box_lists = []
        t = time.time()
        for config in self.scale_configs:
            ystart = config[0]
            ystop = config[1]
            scale = config[2]
            step = config[3]
            box_list = self.extractor.find_cars(image, ystart, ystop, scale, step)
            box_lists.extend(box_list)
        layer1_input_boxes_img, layer1_output_img, layer2_input_boxes_img, layer2_output_img \
            = self.filter.draw_layer_boxes(image, box_lists)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to prediction...')
        return layer1_input_boxes_img, layer1_output_img, layer2_input_boxes_img, layer2_output_img
