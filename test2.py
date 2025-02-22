import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import base64
import os
import time
import threading
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# 🔹 Manually set Google API Key (Replace with your actual key)
GOOGLE_API_KEY = "AIzaSyDmbaqP-Av0pTkjVvzPkzRwE7i0FPCmzFc"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY  

class VehicleDetectionProcessor:
    def __init__(self, video_file, yolo_model_path="yolo12s.pt"):
        """Initializes the vehicle detection processor."""
        self.gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        
        # Load YOLO model
        try:
            self.yolo_model = YOLO(yolo_model_path)
            self.names = self.yolo_model.names
        except Exception as e:
            raise RuntimeError(f"Error loading YOLO model: {e}")

        # Load video file
        self.cap = cv2.VideoCapture(video_file)
        if not self.cap.isOpened():
            raise FileNotFoundError("Error: Could not open video file.")

        # Define ROI (Region of Interest)
        self.area = np.array([(403, 605), (365, 639), (986, 665), (979, 633)], np.int32)

        # Track processed vehicle IDs
        self.processed_track_ids = set()
        self.current_date = time.strftime("%Y-%m-%d")
        self.output_filename = f"vehicle_data_{self.current_date}.txt"
        self.cropped_images_folder = "cropped_vehicles"

        # Ensure output directory exists
        os.makedirs(self.cropped_images_folder, exist_ok=True)

        # Initialize output file if empty
        if not os.path.exists(self.output_filename):
            with open(self.output_filename, "w", encoding="utf-8") as file:
                file.write("Timestamp | Vehicle Type | Vehicle Color | Vehicle Company\n")
                file.write("-" * 80 + "\n")

    def encode_image_to_base64(self, image):
        """Encodes an image to a base64 string."""
        _, img_buffer = cv2.imencode(".jpg", image)
        return base64.b64encode(img_buffer).decode("utf-8")

    def analyze_image_with_gemini(self, image_path):
        """Sends the cropped vehicle image to Gemini AI for analysis."""
        try:
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")

            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Analyze this image and extract only the following details:\n\n"
                                "|Vehicle Type | Vehicle Color | Vehicle Company |\n"
                                "|--------------|--------------|---------------|"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        "description": "Detected vehicle"
                    }
                ]
            )

            response = self.gemini_model.invoke([message])
            return response.content.strip()
        except Exception as e:
            print(f"Error invoking Gemini model: {e}")
            return "Error processing image."

    def process_crop_image(self, image, track_id):
        """Processes a cropped image by saving it and sending it to Gemini AI for analysis."""
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        image_filename = os.path.join(self.cropped_images_folder, f"vehicle_{track_id}_{timestamp}.jpg")
        cv2.imwrite(image_filename, image)  # Save the cropped image

        response_content = self.analyze_image_with_gemini(image_filename)
        extracted_data = response_content.split("\n")[2:]

        # Save extracted vehicle details
        if extracted_data:
            with open(self.output_filename, "a", encoding="utf-8") as file:
                for row in extracted_data:
                    if "--------------" in row or not row.strip():
                        continue
                    values = [col.strip() for col in row.split("|")[1:-1]]
                    if len(values) == 3:
                        vehicle_type, vehicle_color, vehicle_company = values
                        file.write(f"{timestamp} | {vehicle_type} | {vehicle_color} | {vehicle_company}\n")
            print(f"✅ Data saved for track ID {track_id}.")

    def crop_and_process(self, frame, box, track_id):
        """Crops the detected vehicle and processes it."""
        if track_id in self.processed_track_ids:
            print(f"Track ID {track_id} already processed. Skipping.")
            return  

        x1, y1, x2, y2 = map(int, box)
        cropped_image = frame[y1:y2, x1:x2]
        self.processed_track_ids.add(track_id)

        # Process the cropped image in a separate thread
        threading.Thread(target=self.process_crop_image, args=(cropped_image, track_id), daemon=True).start()

    def process_video_frame(self, frame):
        """Processes each frame to detect and track vehicles."""
        frame = cv2.resize(frame, (1090, 800))
        results = self.yolo_model.track(frame, persist=True)

        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [-1] * len(boxes)
            allowed_classes = ["car", "truck"]
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                class_name = self.names[class_id]
                if class_name not in allowed_classes:
                    continue  # Skip if it's not a car or truck
                x1, y1, x2, y2 = map(int, box)

                # Check if vehicle is inside ROI
                if cv2.pointPolygonTest(self.area, (x2, y2), False) >= 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cvzone.putTextRect(frame, f"ID: {track_id}", (x2, y2), 1, 1)
                    cvzone.putTextRect(frame, class_name, (x1, y1), 1, 1)
                    self.crop_and_process(frame, box, track_id)

        return frame
    
    @staticmethod
    def mouse_callback(event, x, y, flags, param):
        """Callback function to print mouse coordinates."""
        if event == cv2.EVENT_MOUSEMOVE:
            print(f"Mouse Position: ({x}, {y})")

    def start_processing(self):
        """Starts processing the video feed."""
        cv2.namedWindow("Vehicle Detection")
        cv2.setMouseCallback("Vehicle Detection", self.mouse_callback)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = self.process_video_frame(frame)
            cv2.polylines(frame, [self.area], True, (0, 255, 0), 2)
            cv2.imshow("Vehicle Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print(f"✅ Data saved to {self.output_filename}")

# Example usage
if __name__ == "__main__":
    video_file = "vidmub.mp4"
    processor = VehicleDetectionProcessor(video_file)
    processor.start_processing()
