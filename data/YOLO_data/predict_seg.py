from ultralytics import YOLO

model = YOLO("runs/segment/train3/weights/best.pt")  # load a custom model

# view results on all test images
results = model.predict(source="images/test", save=True, save_txt=True, conf=0.25, show=True, show_labels=True, show_conf=True, imgsz=1024)  # predict with custom model

for result in results:
    boxes = result.boxes  # get boxes
    masks = result.masks  # get masks
    result.show()  # show results
    result.save(filename="predictions.jpg")  # save results