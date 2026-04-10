import cv2
import numpy as np
from ultralytics import YOLO  # Import YOLO from ultralytics


class ObjectDetector:
    """
    This class defines the object detector for object detection
    using YOLOv8 model for different sources like image, video, and webcam.
    """

    def __init__(self):  # Corrected constructor
        """
        Initialize the model parameters like confidence threshold and class labels.
        Generates random colors for each class to visualize the detections.
        """

        # Confidence threshold for object detection
        self.confidence_threshold = 0.6  

        # List of classes that the YOLO model detects
        self.classes = [
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
            "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
            "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"
        ]

        # Generate random colors for each class for visualization purposes
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))  


    def load_yolo_model(self, model_path):
        """
        Load the YOLOv8 model from the specified path.

        Args:
            model_path (str): The path to the pre-trained YOLOv8 model file.

        Returns:
            bool: True if model is loaded successfully, False otherwise.
        """

        try:
            # Load YOLOv8 model from ultralytics
            self.model = YOLO(model_path)  

            # Print confirmation message
            print("Model loaded successfully!")

        except Exception as e:
            # Print error message if model loading fails
            print(f"Error loading model: {e}")
            return False

        return True


    def detect_objects(self, image):
        """
        Detect objects in the provided image using YOLOv8.

        Args:
            image (numpy array): The image in which to detect objects.

        Returns:
            numpy array: The image with bounding boxes and labels drawn around detected objects.
        """

        # Create a copy of the input image to modify
        original_image = image.copy()  

        # Run inference with the YOLOv8 model
        results = self.model(original_image)  # Run YOLOv8 on the image

        # Extract boxes (coordinates of bounding boxes), scores, and class IDs
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
        scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores
        classes = results[0].boxes.cls.cpu().numpy().astype(int)  # Class IDs

        # Process predictions and draw boxes on the image
        for i in range(len(boxes)):
            if scores[i] > self.confidence_threshold:  # Only consider objects with high confidence

                # Get coordinates and class info
                x1, y1, x2, y2 = boxes[i]
                class_id = classes[i]

                # Generate color for current class
                color = self.colors[class_id % len(self.colors)]

                # Draw bounding box on the image
                cv2.rectangle(original_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                # Draw label on the image
                label = f'{self.classes[class_id]}: {scores[i]:.2f}'
                cv2.putText(original_image, label, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return original_image


    def detect_in_image(self, image_path):
        """
        Detect objects in a single image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            None
        """

        # Read the image from file
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error loading image: {image_path}")
            return None

        # Detect objects in the image
        result = self.detect_objects(image)

        # Display the result in a window
        cv2.imshow('Image Detection Result', result)

        # Wait for Enter key press to proceed to video detection
        print("Press Enter to proceed to video detection...")
        cv2.waitKey(0)  # Wait for any key press

        # Close the image window
        cv2.destroyAllWindows()


    def detect_in_video(self, video_path):
        """
        Detect objects in a video file.

        Args:
            video_path (str): Path to the video file.

        Returns:
            None
        """

        # Open video file
        cap = cv2.VideoCapture(video_path)  

        if not cap.isOpened():
            print("Error: Couldn't open video source")
            return

        print("Press Enter to stop video detection and move to real-time webcam detection.")

        while cap.isOpened():
            # Read each frame of the video
            ret, frame = cap.read()  

            if not ret:
                break

            # Detect objects in the frame
            processed_frame = self.detect_objects(frame)

            # Display the processed frame
            cv2.imshow('Video Detection Result', processed_frame)

            # Check if Enter (ASCII 13) is pressed
            if cv2.waitKey(1) == 13:  # Enter key
                print("Enter pressed. Moving to webcam detection...")
                break

        # Release the video capture object
        cap.release()

        # Close all OpenCV windows
        cv2.destroyAllWindows()


    def detect_in_webcam(self):
        """
        Detect objects in webcam stream.

        Returns:
            None
        """

        # Access the default webcam
        cap = cv2.VideoCapture(0)  

        if not cap.isOpened():
            print("Error: Couldn't access webcam")
            return

        while cap.isOpened():
            # Read each frame from the webcam
            ret, frame = cap.read()  

            if not ret:
                break

            # Detect objects in the frame
            processed_frame = self.detect_objects(frame)

            # Display the processed frame
            cv2.imshow('Webcam Detection Result', processed_frame)

            # Break if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam capture object
        cap.release()

        # Close all OpenCV windows
        cv2.destroyAllWindows()



def main():
    """
    Main function to initialize the ObjectDetector, load the model,
    and run detection on image, video, and webcam.
    """

    # Create an instance of the ObjectDetector class
    detector = ObjectDetector()  

    # Load YOLOv8 model (path to your YOLOv8x model)
    model_path = 'yolov8x.pt'  # YOLOv8 models are saved in .pt format

    # Load model and exit if unsuccessful
    if not detector.load_yolo_model(model_path):
        return

    # 1. Image Detection (First detect in image)
    image_path = 'image.jpg'  # Path to the image file
    detector.detect_in_image(image_path)

    # 2. Video Detection (Then detect in video)
    video_path = 'video.mp4'  # Path to the video file
    detector.detect_in_video(video_path)  # Waits for Enter to continue

    # 3. Webcam Detection (Then open webcam and detect)
    detector.detect_in_webcam()


if __name__ == "__main__":  # Corrected __name__ check

    # Start the program by calling main
    main()
