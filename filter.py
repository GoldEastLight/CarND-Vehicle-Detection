from collections import deque
import numpy as np
from utils import draw_boxes
from scipy.ndimage.measurements import label


class Detector():
    def __init__(self, image_shape):
        self.heatmap = None
        self.labeled_boxes = None
        self.shape = image_shape

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

    def get_labeled_boxes(self, box_list, thresh=2):
        heat = np.zeros(shape=self.shape).astype(np.float)
        heat = self.add_heat(heat, box_list)
        heat = self.apply_threshold(heat, threshold=thresh)
        self.heatmap = np.clip(heat, 0, 255)
        labels = label(self.heatmap)
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


class Filter():
    def __init__(self, image_shape):
        self.boxes = deque(maxlen=30)
        self.detector = Detector(image_shape[:-1])
        self.layer1_threshold = 2
        self.layer2_threshold = 10
        self.layer1_input_boxes = []
        self.layer1_output_boxes = []
        self.layer2_input_boxes = []
        self.layer2_output_boxes = []


    def valid_box(self, box, thresh=64):
        w = thresh
        h = thresh
        area = w * h
        box_w = abs(box[0][0] - box[1][0])
        box_h = abs(box[0][1] - box[1][1])
        box_area = box_w * box_h
        if box_area < area / 2 or box_w < w / 3 or h < h / 3:
            return False
        return True


    def area_filter(self, boxes_list, thresh=64):
        new_boxes_list = []
        w = thresh
        h = thresh
        area = w * h
        for box in boxes_list:
            box_w = abs(box[0][0] - box[1][0])
            box_h = abs(box[0][1] - box[1][1])
            box_area = box_w * box_h
            if box_area < area*0.7 or box_w < w*0.7 or box_h < h*0.7:
                continue
            new_boxes_list.append(box)
        return np.array(new_boxes_list)

    def filter(self, box_lists):
        self.layer1_input_boxes = box_lists
        self.layer1_output_boxes = self.detector.get_labeled_boxes(self.layer1_input_boxes, self.layer1_threshold)
        self.layer1_output_boxes = self.area_filter(self.layer1_output_boxes)

        if self.layer1_output_boxes != []:
            self.boxes.append(self.layer1_output_boxes)
        boxes_len = len(self.boxes)
        if boxes_len > 0:
            self.layer2_input_boxes = np.concatenate(np.array(self.boxes))
        else:
            self.layer2_input_boxes = []
        layer2_threshold = min(boxes_len, self.layer2_threshold)
        self.layer2_output_boxes = self.detector.get_labeled_boxes(self.layer2_input_boxes, layer2_threshold)
        self.layer2_output_boxes = self.area_filter(self.layer2_output_boxes)

    def draw_layer_boxes(self, image, box_lists):
        self.filter(box_lists)
        layer1_input_boxes_img = draw_boxes(image, self.layer1_input_boxes)
        layer1_output_img = draw_boxes(image, self.layer1_output_boxes)
        layer2_input_boxes_img = draw_boxes(image, self.layer2_input_boxes)
        layer2_output_img = draw_boxes(image, self.layer2_output_boxes)
        return layer1_input_boxes_img, layer1_output_img, layer2_input_boxes_img, layer2_output_img
