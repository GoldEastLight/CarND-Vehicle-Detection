from collections import deque
import numpy as np
from utils import draw_boxes, colors, draw_namebox, topmost
from scipy.ndimage.measurements import label
from collections import namedtuple
import time
from extract import FeatureExtractor
import cv2
from train import train_parameters, model

filter_parameters = {
    'confidence_threshold':40,
    'dir_threshold':2,
    'edge_margin':100,
    'increase_data_threshold':5,
    'layer1_threshold': 1,
    'layer2_threshold': 10,
    'valid_size_scale_threshold': 2,
    'box_w_h_scale_threshold': 5
}


class Detector():
    def __init__(self, image_shape):
        self.heatmap = None
        self.labeled_boxes = None
        self.shape = image_shape
        self.max_heat = 0
        self.mean_heat = 0

    def add_heat(self, heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        # Return updated heatmap
        return heatmap  # Iterate through list of bboxes

    def apply_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

    def draw_labeled_boxes(self, image, box_list, thresh=2):
        labeled_boxes = self.get_labeled_boxes(box_list, thresh)
        labeled_boxes_img = draw_boxes(image, labeled_boxes)
        return labeled_boxes_img

    def get_labeled_boxes(self, box_list, thresh=2, structure = None):
        heat = np.zeros(shape=self.shape).astype(np.float)
        heat = self.add_heat(heat, box_list)
        self.max_heat, self.mean_heat = topmost(heat, 200, "headmap")
        heat = self.apply_threshold(heat, threshold=thresh)
        self.heatmap = np.clip(heat, 0, 255)
        labels = label(self.heatmap, structure)
        self.heatmap = self.heatmap.astype(np.uint8)
        labeled_boxes = []
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            box = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            labeled_boxes.append(box)
        self.labeled_boxes = np.array(labeled_boxes)
        return self.labeled_boxes


class Vehicle():
    def __init__(self, color, name, box):
        self.boxes = deque(maxlen=10)
        self.confidence = 0
        self.direction = 'K' # 'F': far 'N': Near 'K': keep
        self.display = False
        self.avg_box = None
        self.left_x = 640
        self.right_x = 640
        self.dir_threshold = filter_parameters['dir_threshold']
        self.edge_margin = filter_parameters['edge_margin']
        self.color = color
        self.name = name
        self.decrease = 0
        self.increase_data = False
        self.updated = False
        self.update(box)

    def update(self, box):
        self.updated = True
        if box is not None:
            self.boxes.append(box)
            self.direction = self.get_direction()
            if len(self.boxes) > 0:
                sum = 0
                sum_box = np.zeros_like(self.boxes[0])
                for i, box in enumerate(np.array(self.boxes)):
                    sum += i + 1
                    weight = i + 1
                    sum_box += box * weight
                self.avg_box = (sum_box / sum).astype(np.int)
                self.left_x = self.avg_box[0][0]
                self.right_x = self.avg_box[1][0]
            else:
                self.avg_box = None
            self.cal_confidence(True)
        else:
            if len(self.boxes) > 0:
                self.boxes.popleft()
            self.cal_confidence(False)
        self.cal_display()

        return self.display

    def deleted(self):
        if self.confidence < 0:
            return True
        else:
            return False

    def clean(self):
        self.boxes.clear()

    def cal_display(self):
        if self.confidence > 0:
            self.display = True
        else:
            self.display = False
        return self.display

    def get_direction(self):
        box_list = np.array(self.boxes)
        if len(box_list) < self.dir_threshold:
            return 'K'
        box = box_list[0]
        last_center_y = (box[0][1] + box[1][1]) / 2
        dir_cnt = 0
        for box in box_list[1:]:
            center_y = (box[0][1] + box[1][1]) / 2
            if center_y > last_center_y: # N
                dir_cnt += 1
            elif center_y < last_center_y: # F
                dir_cnt -= 1
            last_center_y = center_y # K
        if dir_cnt > 0:
            return 'N'
        elif dir_cnt < 0:
            return 'F'
        else:
            return 'K'

    def cal_confidence(self, find_car=False):
        if self.left_x - 0 < self.edge_margin or 1280 - self.right_x < self.edge_margin:
            self.confidence += 1 if find_car else -max(np.ceil(self.confidence / 2).astype(int), 1)
        else:
            self.confidence += 1 if find_car else -1
        if find_car:
            self.decrease -= 1 if self.decrease > 0 else 0
        else:
            self.decrease += 1
        if self.decrease > filter_parameters['increase_data_threshold']:
            self.increase_data = True
        else:
            self.increase_data = False

        self.confidence = min(self.confidence, filter_parameters['confidence_threshold'])
        return self.confidence


class Filter():
    def __init__(self, image_shape):
        self.boxes = deque(maxlen=30)
        self.detector = Detector(image_shape[:-1])
        self.layer1_heatmap = None
        self.layer2_heatmap = None
        self.layer1_threshold = filter_parameters['layer1_threshold']
        self.layer2_threshold = filter_parameters['layer2_threshold']
        self.layer1_input_boxes = []
        self.layer1_output_boxes = []
        self.layer2_input_boxes = []
        self.layer2_output_boxes = []
        self.vehicles = []
        self.n_car = 0
        self.layer1_max_heat = 0
        self.layer2_max_heat = 0
        self.increase_data = False

    def cal_overlap_ratio(self, vehicle_box, detected_box):
        Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
        v_b = Rectangle(vehicle_box[0][0], vehicle_box[0][1], vehicle_box[1][0], vehicle_box[1][1])
        d_b = Rectangle(detected_box[0][0], detected_box[0][1], detected_box[1][0], detected_box[1][1])
        dx = min(v_b.xmax, d_b.xmax) - max(v_b.xmin, d_b.xmin)
        dy = min(v_b.ymax, d_b.ymax) - max(v_b.ymin, d_b.ymin)
        v_b_area = (v_b.xmax - v_b.xmin) * (v_b.ymax - v_b.ymin)
        d_b_area = (d_b.xmax - d_b.xmin) * (d_b.ymax - d_b.ymin)
        # min_area = min(v_b_area, d_b_area)
        if (dx >= 0) and (dy >= 0):
            area = dx*dy
            ratio = area #(area / v_b_area + area / d_b_area) / 2
        else:
            ratio = 0
        size_scale = d_b_area / v_b_area
        return ratio, size_scale

    def find_max_overlap_ratio(self, box):
        max_overlap_ratio = 0
        overlap_cnt = 0
        max_vehicle_idx = 0
        for vehicle_idx in range(len(self.vehicles)):
            overlap_ratio, _ = self.cal_overlap_ratio(self.vehicles[vehicle_idx].avg_box, box)
            overlap_cnt += 1 if overlap_ratio > 0 else 0
            if overlap_ratio > max_overlap_ratio:
                max_overlap_ratio = overlap_ratio
                max_vehicle_idx = vehicle_idx
        return max_overlap_ratio, max_vehicle_idx

    def update_box(self, boxes, increased_boxes):

        for v_i in range(len(self.vehicles)):
            max_overlap_ratio = 0
            win_box_idx = 0
            overlap_cnt = 0
            if self.vehicles[v_i].updated:
                continue
            for b_i, box in enumerate(boxes):
                overlap_ratio, _ = self.cal_overlap_ratio(self.vehicles[v_i].avg_box, box)
                overlap_cnt += 1 if overlap_ratio > 0 else 0
                if overlap_ratio > max_overlap_ratio:
                    max_overlap_ratio = overlap_ratio
                    win_box_idx = b_i
            if max_overlap_ratio > 0:
                ratio, max_v_i = self.find_max_overlap_ratio(boxes[win_box_idx])
                vehicle_idx = v_i
                if max_v_i != v_i and ratio > max_overlap_ratio:
                    vehicle_idx = max_v_i
                if overlap_cnt > 1:
                    self.vehicles[vehicle_idx].clean()
                self.vehicles[vehicle_idx].update(boxes[win_box_idx])
                boxes = np.delete(boxes, win_box_idx, axis=0)

        self.increase_data = False
        for v_i in range(len(self.vehicles)):
            max_overlap_ratio = 0
            win_box_idx = 0
            valid_size_scale = 0
            if self.vehicles[v_i].increase_data:
                self.increase_data = True
                for b_i, box in enumerate(increased_boxes):
                    overlap_ratio, size_scale = self.cal_overlap_ratio(self.vehicles[v_i].avg_box, box)
                    if overlap_ratio > max_overlap_ratio:
                        max_overlap_ratio = overlap_ratio
                        win_box_idx = b_i
                        valid_size_scale = size_scale
                if max_overlap_ratio > 0 and valid_size_scale < filter_parameters['valid_size_scale_threshold']:
                    ratio, max_v_i = self.find_max_overlap_ratio(increased_boxes[win_box_idx])
                    if v_i == max_v_i:
                        self.vehicles[v_i].update(increased_boxes[win_box_idx])

        if len(boxes) > 0:
            for box in boxes:
                self.n_car += 1
                color = colors[self.n_car%len(colors)]
                name = 'Car' + str(self.n_car)
                self.vehicles.append(Vehicle(color, name, box))

        for v_i in range(len(self.vehicles)):
            if self.vehicles[v_i].updated:
                self.vehicles[v_i].updated = False
            else:
                self.vehicles[v_i].update(None)
                self.vehicles[v_i].updated = False

        self.vehicles = [vehicle for vehicle in self.vehicles if not vehicle.deleted()]

    def draw_vehicles(self, img):
        imcopy = np.copy(img)
        for vehicle in self.vehicles:
            if vehicle.display:
                color = vehicle.color
                name = vehicle.name + ':' + vehicle.direction + str(vehicle.confidence)
                box = vehicle.avg_box
                draw_namebox(imcopy, box, name, color)
        return imcopy

    def area_filter(self, boxes_list, thresh=64):
        # return boxes_list
        new_boxes_list = []
        w = thresh
        h = thresh
        area = w * h
        for box in boxes_list:
            box_w = abs(box[0][0] - box[1][0])
            box_h = abs(box[0][1] - box[1][1])
            box_area = box_w * box_h
            scale = filter_parameters['box_w_h_scale_threshold']
            if box_w / box_h > scale or box_h / box_w > scale or box_area < area:
                continue
            new_boxes_list.append(box)
        return np.array(new_boxes_list)

    def filter(self, box_lists):
        self.layer1_input_boxes = box_lists
        self.layer1_output_boxes = self.detector.get_labeled_boxes(self.layer1_input_boxes, self.layer1_threshold, None)
        self.layer1_heatmap = self.detector.heatmap
        self.layer1_max_heat = self.detector.max_heat
        # self.layer1_output_boxes = self.area_filter(self.layer1_output_boxes)

        if self.layer1_output_boxes != []:
            self.boxes.append(self.layer1_output_boxes)
        else:
            self.boxes.append([[[0,0], [0,0]]])
        boxes_len = len(self.boxes)
        if boxes_len > 0:
            self.layer2_input_boxes = np.concatenate(np.array(self.boxes))
        else:
            self.layer2_input_boxes = []

        layer2_threshold = min(boxes_len, self.layer2_threshold)
        print('layer2_threshold', layer2_threshold, boxes_len, self.layer2_max_heat)
        structure = [[1,1,1],[1,1,1],[1,1,1]]
        self.layer2_output_boxes = self.detector.get_labeled_boxes(self.layer2_input_boxes, layer2_threshold, structure)
        self.layer2_max_heat = self.detector.max_heat
        self.layer2_heatmap = self.detector.heatmap
        self.layer2_output_boxes = self.area_filter(self.layer2_output_boxes)
        self.update_box(self.layer2_output_boxes, self.layer1_output_boxes)

    def draw_layer_boxes(self, image, box_lists):
        self.filter(box_lists)
        layer1_input_boxes_img = draw_boxes(image, self.layer1_input_boxes)
        layer1_output_img = draw_boxes(image, self.layer1_output_boxes)
        layer2_input_boxes_img = draw_boxes(image, self.layer2_input_boxes)
        layer2_output_img = draw_boxes(image, self.layer2_output_boxes)
        vehicles_img = self.draw_vehicles(image)
        return layer1_input_boxes_img, layer1_output_img, \
               layer2_input_boxes_img, layer2_output_img, \
               self.layer1_heatmap, self.layer2_heatmap, \
               vehicles_img


class Tracker():
    def __init__(self, parameters=train_parameters, model=model):
        self.extractor = FeatureExtractor(parameters, model)
        self.filter = Filter((720, 1280, 3))
        self.scale_configs = (
            (410, 480, 0.8, 1),
            (400, 550, 1.1, 1),
            (400, 620, 1.5, 2),
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
            if self.filter.layer1_max_heat > 5 and not self.filter.increase_data:
                step = 2
            box_list, slide_boxes = self.extractor.find_cars(image, ystart, ystop, scale, step)
            slide_boxes_list.append(slide_boxes)
            box_lists.extend(box_list)

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