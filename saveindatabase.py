import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import base64
import os
import time
import threading
import mysql.connector
from mysql.connector import Error
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Manually set Google API Key (Replace with your actual key)
GOOGLE_API_KEY = ""
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY  

class VehicleDetectionProcessor:
    def __init__(self, video_file, yolo_model_path="yolo12n.pt"):
        """Initializes the vehicle detection processor with MySQL database connection."""
        self.gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

        try:
            self.yolo_model = YOLO(yolo_model_path)
            self.names = self.yolo_model.names
        except Exception as e:
            raise RuntimeError(f"Error loading YOLO model: {e}")

        self.cap = cv2.VideoCapture(video_file)
        if not self.cap.isOpened():
            raise FileNotFoundError("Error: Could not open video file.")

        self.area = np.array([(392, 407), (382, 448), (740, 456), (734, 419)], np.int32)
        self.processed_track_ids = set()
        self.cropped_images_folder = "cropped_vehicles"
        os.makedirs(self.cropped_images_folder, exist_ok=True)

        # MySQL Database Connection
        self.db_connection = self.connect_to_database()
        self.create_table_if_not_exists()

    def connect_to_database(self):
        """Connects to the MySQL database (creates it if not exists)."""
        try:
            connection = mysql.connector.connect(
                host="localhost",
                user="root",   # Change if you have a different MySQL user
                password=""    # Default XAMPP MySQL has no password
            )
            cursor = connection.cursor()
            cursor.execute("CREATE DATABASE IF NOT EXISTS vehicle_detection")
            connection.database = "vehicle_detection"
            print("✅ Connected to MySQL database.")
            return connection
        except Error as e:
            print(f"❌ Error connecting to MySQL: {e}")
            return None

    def create_table_if_not_exists(self):
        """Creates the vehicles table if it doesn't exist."""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vehicles (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    timestamp VARCHAR(50),
                    track_id INT,
                    vehicle_type VARCHAR(50),
                    vehicle_color VARCHAR(50),
                    vehicle_company VARCHAR(50)
                )
            """)
            self.db_connection.commit()
            print("✅ Table `vehicles` ready.")
        except Error as e:
            print(f"❌ Error creating table: {e}")

    def save_data_to_mysql(self, timestamp, track_id, vehicle_type, vehicle_color, vehicle_company):
        """Inserts detected vehicle data into the MySQL database."""
        try:
            cursor = self.db_connection.cursor()
            query = "INSERT INTO vehicles (timestamp, track_id, vehicle_type, vehicle_color, vehicle_company) VALUES (%s, %s, %s, %s, %s)"
            values = (timestamp, track_id, vehicle_type, vehicle_color, vehicle_company)
            cursor.execute(query, values)
            self.db_connection.commit()
            print(f"✅ Data saved for track ID {track_id} in MySQL.")
        except Error as e:
            print(f"❌ Error saving data to MySQL: {e}")

    def analyze_image_with_gemini(self, image_path):
        """Sends an image to Gemini AI for analysis and extracts vehicle details."""
        try:
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")

            message = HumanMessage(
                content=[
                    {"type": "text", "text": "Analyze this image and extract only the following details:\n\n"
                     "|Vehicle Type(Name of Vehicle) | Vehicle Color | Vehicle Company |\n"
                     "|--------------|--------------|---------------|"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, "description": "Detected vehicle"}
                ]
            )

            response = self.gemini_model.invoke([message])
            return response.content.strip()
        except Exception as e:
            print(f"❌ Error invoking Gemini model: {e}")
            return "Error processing image."

    def process_crop_image(self, image, track_id):
        """Processes a cropped vehicle image and saves data to MySQL."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        image_filename = os.path.join(self.cropped_images_folder, f"vehicle_{track_id}_{timestamp.replace(':', '-')}.jpg")
        cv2.imwrite(image_filename, image)

        response_content = self.analyze_image_with_gemini(image_filename)
        extracted_data = response_content.split("\n")[2:]

        if extracted_data:
            for row in extracted_data:
                if "--------------" in row or not row.strip():
                    continue
                values = [col.strip() for col in row.split("|")[1:-1]]
                if len(values) == 3:
                    vehicle_type, vehicle_color, vehicle_company = values
                    self.save_data_to_mysql(timestamp, track_id, vehicle_type, vehicle_color, vehicle_company)

    def crop_and_process(self, frame, box, track_id):
        """Crops the vehicle from the frame and sends it for analysis."""
        if track_id in self.processed_track_ids:
            print(f"Track ID {track_id} already processed. Skipping.")
            return  

        x1, y1, x2, y2 = map(int, box)
        cropped_image = frame[y1:y2, x1:x2]
        self.processed_track_ids.add(track_id)
        threading.Thread(target=self.process_crop_image, args=(cropped_image, track_id), daemon=True).start()

    def process_video_frame(self, frame):
        """Processes each video frame and detects vehicles using YOLO."""
        frame = cv2.resize(frame, (1020, 600))
        results = self.yolo_model.track(frame, persist=True)

        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [-1] * len(boxes)
            allowed_classes = ["car", "truck"]
            
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                class_name = self.names[class_id]
                if class_name not in allowed_classes:
                    continue
                x1, y1, x2, y2 = map(int, box)
                if cv2.pointPolygonTest(self.area, (x2, y2), False) >= 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cvzone.putTextRect(frame, f"ID: {track_id}", (x2, y2), 1, 1)
                    cvzone.putTextRect(frame, class_name, (x1, y1), 1, 1)
                    self.crop_and_process(frame, box, track_id)
        return frame

    def start_processing(self):
        """Starts the video processing loop."""
        cv2.namedWindow("Vehicle Detection")
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
        print("✅ Processing completed.")

if __name__ == "__main__":
    video_file = "my.mp4"
    processor = VehicleDetectionProcessor(video_file)
    processor.start_processing()
