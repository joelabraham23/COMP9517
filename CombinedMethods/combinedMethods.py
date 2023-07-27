import json
import sys
import cv2
import matplotlib.pyplot as plt
import os

sys.path.insert(0, "../Yolov8")
sys.path.insert(1, "../cannySiftNN")
sys.path.insert(2, "../color")

from YoloClassify import main as yolo_main
from cannySiftNN_train import main as canny_sift_main
from densenet import DenseNet
from color_model import predict_image_category
from fClassMachine import main as fClassMain


def process_image(image_path):
    yolo_label, yolo_score, yolo_box = yolo_main(image_path)
    canny_sift_label, canny_sift_score = canny_sift_main(image_path)
    color_label, color_proba = predict_image_category(image_path)
    fClass_label, fClass_proba = fClassMain(image_path)

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
        "color": {
            "label": color_label,
            "score": color_proba,
        },
        "Feature Classifier": {
            "label": fClass_label,
            "score": fClass_proba,
        },
    }


def weighted_result(result_json):
    weights = {"yolo": 0.50, "cannySiftNN": 0.20, "color": 0.10, "fClass": 0.20}
    scores = {"penguin": 0.0, "turtle": 0.0}
    for method in result_json.keys():
        scores[result_json[method]["label"]] += (
            result_json[method]["score"] * weights[method]
        )
    predicted = max(scores, key=scores.get)
    return predicted, scores[predicted]


def main(image_path):
    # Classification result
    result = process_image(image_path)
    final_result = weighted_result(result)

    # Draw boundiong box on image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x1, y1, x2, y2 = result["yolo"]["box"]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Draw classification result on image
    text = final_result[0]
    position = (x1, y1 - 50)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    text = str(round(final_result[1], 2))
    position = (x1, y1 - 10)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    plt.imshow(image)
    plt.show()
    return final_result, image


main("./valid/image_id_000.jpg")
