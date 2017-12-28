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
from filter import Filter
from utils import draw_boxes, colors

train_parameters = {
    'c_color_space': 'YUV',
    'h_color_space': 'YUV',
    'orient': 10,
    'pix_per_cell': 8,
    'cell_per_block': 4,
    'hog_channel': 6, #3 0+1 # 4 0+2 # 5 1 + 2
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
    # sample_size = 100
    cars = vehicles_files
    notcars = non_vehicles_files

    # # sample_size = 100
    # cars = vehicles_files[0:sample_size]
    # notcars = non_vehicles_files[0:sample_size]


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


class Tracker():
    def __init__(self, parameters=train_parameters, model=model):
        self.extractor = FeatureExtractor(parameters, model)
        self.filter = Filter((720, 1280, 3))
        self.scale_configs = (
            (410, 480, 1, 1),
            (390, 620, 1.2, 2),
            (400, 620, 1.5, 2),
            # (550, 650, 1.6, 2),
            # (500, 720, 2, 1),
            # (400, 700, 3, 2),
        )

    def detect_car(self, image):
        box_lists = []
        slide_boxes_list = []
        t = time.time()
        for i, config in enumerate(self.scale_configs):
            ystart = config[0]
            ystop = config[1]
            scale = config[2]
            step = config[3]
            box_list, slide_boxes = self.extractor.find_cars(image, ystart, ystop, scale, step)
            slide_boxes_list.append(slide_boxes)
            box_lists.extend(box_list)
            # if i == 0 and len(box_list) < 2:
            #     ystart, ystop, scale, step = 410, 470, 0.7, 1
            #     box_list, slide_boxes = self.extractor.find_cars(image, ystart, ystop, scale, step)
            #     slide_boxes_list.append(slide_boxes)
            #     box_lists.extend(box_list)
        layer1_input_boxes_img, layer1_output_img, \
        layer2_input_boxes_img, layer2_output_img, \
        layer1_heatmap, layer2_heatmap, vehicles_img \
            = self.filter.draw_layer_boxes(image, box_lists)
        slide_boxes_img = image
        for i, slide_boxes in enumerate(slide_boxes_list):
            color_idx = i % len(colors)
            slide_boxes_img = draw_boxes(slide_boxes_img, slide_boxes, color=colors[color_idx], thick=1, colorful=True)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to prediction...')
        layer1_heatmap = ((layer1_heatmap / (np.max(layer1_heatmap) + 1)) * 255).astype(np.uint8)
        layer2_heatmap = ((layer2_heatmap / (np.max(layer2_heatmap) + 1)) * 255).astype(np.uint8)
        layer1_heatmap = cv2.applyColorMap(layer1_heatmap, cv2.COLORMAP_HOT)
        layer2_heatmap = cv2.applyColorMap(layer2_heatmap, cv2.COLORMAP_HOT)
        return layer1_input_boxes_img, layer1_output_img, \
               layer2_input_boxes_img, layer2_output_img, \
               layer1_heatmap, layer2_heatmap, \
               slide_boxes_img, vehicles_img
