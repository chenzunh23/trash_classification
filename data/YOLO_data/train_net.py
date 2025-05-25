from ultralytics import YOLO
import argparse
import format_yolo_data
from predict_seg import predict_seg_model

def train_seg_model(model="yolo11n-seg.pt", epochs = 150, imgsz = 600, batch = 8, workers = 4, device = 0):
    """
    Training YOLOv8 segmentation models

    requires datasets be split into train, val, and test folders
    """
    # Load segmentation model
    model = YOLO(model)  # load an official detection model

    model.info()  # print model information
    # Predict with segmentation model
    model.train(data = "data.yaml", epochs = epochs, imgsz = imgsz, batch = batch, workers = workers, device = device)  # train the model

    model.export()

    # Validate the model
    metrics = model.val()  # validate the model
    metrics.box.map  # get mAP metric
    metrics.box.map50  # map50(B)
    metrics.box.map75  # map75(B)
    metrics.box.maps  # a list contains map50-95(B) of each category
    metrics.seg.map  # map50-95(M)
    metrics.seg.map50  # map50(M)
    metrics.seg.map75  # map75(M)
    metrics.seg.maps  # a list contains map50-95(M) of each category

def test_model(model="yolo11n-seg.pt", data = "data.yaml"):
    model = YOLO(model)  # load an official detection model
    model.info()  # print model information
    model.predict('./images/test', save=True, save_txt=True, show=False, conf=0.25, show_labels=True, show_conf=False, imgsz=640)  # predict with the model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 segmentation model")
    parser.add_argument("--model", default="yolo11n-seg.pt", type=str, help="Path to the YOLOv8 segmentation model")
    parser.add_argument("--data", default="data.yaml", type=str, help="Path to the dataset YAML file")
    parser.add_argument("--epochs", default=150, type=int, help="Number of epochs to train")
    parser.add_argument("--imgsz", default=600, type=int, help="Image size for training")
    parser.add_argument("--batch", default=8, type=int, help="Batch size for training")
    parser.add_argument("--workers", default=4, type=int, help="Number of workers for data loading")
    parser.add_argument("--device", default=0, type=int, help="Device to use for training (0 for GPU, -1 for CPU)")
    parser.add_argument("--format", default="", help="Format the dataset again using input format")
    parser.add_argument("--test", action="store_true", help="Test a model after training")
    args = parser.parse_args()
    if args.format != "":
        format_yolo_data.format_yolo_data(args.format)
    train_seg_model(args.model, args.epochs, args.imgsz, args.batch, args.workers, args.device)
    if args.test:
        test_model(args.model, args.data)
    