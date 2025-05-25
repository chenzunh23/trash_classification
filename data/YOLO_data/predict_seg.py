from ultralytics import YOLO

# Avoid import problems
import sys
from pathlib import Path

current_dir = Path(__file__).parent

sys.path.append(str(current_dir))
from format_yolo_data import yolo_to_taco
import os
import argparse

def predict_seg_model(model, source, save=True, save_txt=True, show=False, conf=0.25, show_labels=True, show_conf=True, imgsz=640):
    """
    Predict with YOLOv8 segmentation models
    """

    # view results on all test images
    results = model.predict(source=source, save=save, save_txt=save_txt, show=show, conf=conf, show_labels=show_labels, show_conf=show_conf, imgsz=imgsz)
    # Predict with segmentation model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict with YOLOv8 segmentation model")
    parser.add_argument("--model", default="yolo11n-seg.pt", type=str, help="Path to the YOLOv8 segmentation model")
    parser.add_argument("--source", default="../../GroupedPhotos/Group_5/5_1.jpg", type=str, help="Path to the image or video file (or files)")
    parser.add_argument("--no_save", action="store_false", help="Don't save the results")
    parser.add_argument("--no_save_txt", action="store_false", help="Don't save the results as YOLO label files")
    parser.add_argument("--show", action="store_true", help="Show the results")
    parser.add_argument("--conf", default=0.25, type=float, help="Confidence threshold")
    parser.add_argument("--no_labels", action="store_false", help="Show labels on the results")
    parser.add_argument("--show_conf", action="store_true", help="Show confidence on the results")
    parser.add_argument("--imgsz", default=640, type=int, help="Image size for prediction")
    parser.add_argument("--format", action="store_true", help="Format the dataset again using input format")
    args = parser.parse_args()
    model = YOLO(args.model)  # load an official detection model
    # Predict with segmentation model
    predict_seg_model(model=model, source=args.source, save=args.no_save, save_txt=args.no_save_txt, show=args.show, conf=args.conf, show_labels=args.no_labels, show_conf=args.show_conf, imgsz=args.imgsz)
    # Convert YOLO labels back to TACO format
    # Get the current predict directory
    if args.format:
        predict_cnt = 0
        for i in os.listdir("./runs/segment/"):
            if "predict" in i:
                predict_cnt += 1
        print(f"Predict count: {predict_cnt}")
        predict_dir = f"./runs/segment/predict{predict_cnt}/labels"
        yolo_to_taco("", predict_dir, f"output{predict_cnt}.json")