from ultralytics import YOLO

# Load segmentation model
model = YOLO("yolo11n-seg.pt")  # load an official detection model

model.info()  # print model information
# Predict with segmentation model
model.train(data = "data.yaml", epochs = 50, imgsz = 600, batch = 4, device = 0)  # train the model

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