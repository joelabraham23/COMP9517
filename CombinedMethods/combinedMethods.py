# combined.py
import sys
sys.path.insert(0, '../Yolov8')
sys.path.insert(1, '../cannySiftNN')
sys.path.insert(2, '../color')

from YoloClassify import main as yolo_main
from cannySiftNN_train import main as canny_sift_main
from densenet import DenseNet
#from color_model import predict_image_category

def process_image(image_path):
    yolo_label, yolo_score, yolo_box = yolo_main(image_path)
    canny_sift_label,canny_sift_score = canny_sift_main(image_path)
    #color_label, color_proba = predict_image_category(image_path)

    return {
        "yolo": {
            "label": yolo_label,
            "score": yolo_score,
            "box": yolo_box,
        },
        "cannySiftNN": {
            "label": canny_sift_label,
            "score": canny_sift_score,
        },
        #"color": {
        #    "label": color_label,
        #    "score": color_proba,
        #}
    }

# Example usage
result = process_image("./valid/valid/image_id_000.jpg")
print(result)
