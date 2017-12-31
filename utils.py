import cv2
import numpy as np

colors = [(255, 255, 0), (0,255,0), (0,0,255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 128, 128), (128, 255, 0), (0, 255, 128)]


def draw_namebox(img, box, name, color, thick=3):
    x = box[0][0]
    y = box[0][1] - 10
    cv2.putText(img, text=name, org=(x, y), fontFace=cv2.FONT_HERSHEY_PLAIN, \
                fontScale=2, color=color, thickness=2, lineType=cv2.LINE_AA)
    cv2.rectangle(img, tuple(box[0]), tuple(box[1]), color, thick)

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=3, colorful=False):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    default_thick = thick
    try:
        for i, bbox in enumerate(bboxes):
            if colorful:
                color = colors[i%len(colors)]
                if i == 0 or i == len(bboxes)-1:
                    thick = 3
                    color = (255,255,255)
                else:
                    thick = default_thick
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, tuple(bbox[0]), tuple(bbox[1]), color, thick)
    except:
        print('Error:', bboxes)
    # Return the image copy with boxes drawn
    return imcopy

def show_line(img, line):
    x = 10
    y = 670
    cv2.putText(img, text=line, org=(x, y), fontFace=cv2.FONT_HERSHEY_PLAIN, \
                fontScale=3, color=(0, 255, 255), thickness=3, lineType=cv2.LINE_AA)

def resize_image(img, shape, title = ''):
    if len(img.shape) == 3:
        stack = img
    else:
        stack = np.dstack((img, img, img))
    show_line(stack, title)
    resize_img = cv2.resize(stack, shape, interpolation=cv2.INTER_CUBIC)
    h = resize_img.shape[0]-1
    w = resize_img.shape[1]-1
    cv2.rectangle(resize_img, (0, 0), (w, h), (0, 0, 255), 1)
    return resize_img
