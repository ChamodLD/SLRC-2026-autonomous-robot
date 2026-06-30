from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO(r"C:\Users\USER\runs\detect\train2\weights\best.pt")

# -----------------------------
# OPTION 1: Test on single image
# -----------------------------
def test_image(image_path):
    results = model(image_path, conf=0.25, save=True)
    print("Detection complete. Results saved in runs/detect/predict")

# -----------------------------
# OPTION 2: Test on folder
# -----------------------------
def test_folder(folder_path):
    results = model(folder_path, conf=0.25, save=True)
    print("Folder detection complete.")

# -----------------------------
# OPTION 3: Real-time webcam
# -----------------------------
def test_webcam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.25)

        annotated_frame = results[0].plot()

        cv2.imshow("YOLOv8 Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 🔹 Uncomment ONE option at a time

    # Test single image
    # test_image("test.jpg")

    # Test folder
    # test_folder(r"C:\Users\USER\Box_detection.v2i.yolov8\test\images")

    # Test webcam
    test_webcam()
