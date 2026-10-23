from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.export(
	format = "ncnn",
	imgsz = 320
)
print("YOLO NCNN export complete!")
