import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import os
import time
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler('speed_violation_detection.log')])


def process_frame(frame, upper_roi_y, lower_roi_y, offset, car_info, line_distance_met, speed_limit_m, output_folder):
    logging.info("Processing frame...")  # Log: Indicate frame processing start
    # Create a copy of the frame for output
    output_image = frame.copy()

    # Draw lines defining regions of interest
    cv2.line(output_image, (370, upper_roi_y), (710, upper_roi_y), (255, 255, 255), 1)
    cv2.line(output_image, (660, lower_roi_y), (1290, lower_roi_y), (255, 255, 255), 1)

    # Detect common objects in the frame using YOLOv3
    bbox, label, conf = cv.detect_common_objects(frame, confidence=0.5, model='yolov3')
    draw_bbox(output_image, bbox, label, conf)

    # List to store objects to be removed from car_info
    objects_to_remove = []

    # Define the color and font for the speed text label
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_color = (0, 0, 255)  # Blue color for the speed text label

    # Loop through detected objects
    for i, obj_label in enumerate(label):
        # Calculate the center of the bounding box
        x_center, y_center = [(int(bbox[i][j]) + int(bbox[i][j + 2])) // 2 for j in range(2) if bbox[i][j] is not None]
        obj_key = f"{obj_label}-{i}"  # Using 'i' as a unique identifier

        # Update the bounding box information in car_info
        if obj_key not in car_info:
            car_info[obj_key] = {'bbox': bbox[i], 'enter_time': time.time(), 'crossed_upper_roi': False,
                                'speeding': False, 'has_exited': False, 'speeding_image_saved': False}
            logging.debug("update the bounding box info") # debugging print

        # Check if the object is within the region of interest
        if (upper_roi_y - offset) < y_center < (lower_roi_y + offset):
            logging.debug("check if the object is within the ROI") # Log: Check object within ROI
            if not car_info[obj_key]['crossed_upper_roi']:
                # Mark that the object has crossed upper_roi
                car_info[obj_key]['crossed_upper_roi'] = True
        elif car_info[obj_key]['crossed_upper_roi']:
            logging.debug("check if the object has crossed the upper roi") # Log: Check object crossed upper ROI
            # Inside the loop where you check if the object has exited the region
            logging.debug("Checking if the object has exited the region")  # Log: Checking if object exited region
            if y_center < (upper_roi_y - offset) or y_center > (lower_roi_y + offset):
                logging.debug("Object has exited the region")  # Log: Object has exited the region
                # Check if the detected speed exceeds the speed limit
                time_elapsed = time.time() - car_info[obj_key]['enter_time']
                if time_elapsed > 0:
                    detected_speed = line_distance_met / time_elapsed * 2.23694  # Convert to mph
                    logging.debug(f"Detected Speed for {obj_key}: {detected_speed:.1f} mph")  # Log: Detected speed

                    speed_limit = speed_limit_m  # Setting the predetermined speed limit

                    if detected_speed > speed_limit:
                        logging.info(f"Object is speeding (Detected Speed: {detected_speed:.1f} mph, Speed Limit: {speed_limit} mph)")  # Log: Object is speeding

                        # Save images for speeding violation only once
                        if not car_info[obj_key]['speeding_image_saved']:
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            image_path = os.path.join(output_folder, f'speeding_incident_{timestamp}_{obj_key}.jpg')
                            cv2.imwrite(image_path, output_image)
                            logging.debug(f"Saved speeding incident image: {image_path}")  # Log: Saved speeding incident image

                            car_info[obj_key]['speeding_image_saved'] = True

                        objects_to_remove.append(obj_key)
                    else:
                        logging.debug("Object is not speeding")  # Log: Object is not speeding
            else:
                # Calculate the speed and display it next to the object
                time_elapsed = time.time() - car_info[obj_key]['enter_time']
                if time_elapsed > 0:
                    mph = line_distance_met / time_elapsed * 2.23694
                    speed_text = f"{mph:.1f} mph"
                    logging.debug(f"Speed for {obj_key}: {speed_text}")  # Log: Speed for the object

                    # Update the speed text position based on the object's bounding box
                    text_position = (int(x_center - 20), int(y_center - 10))
                    cv2.putText(output_image, speed_text, text_position, font, font_scale, text_color, font_thickness)

    # Remove objects that have exited from car_info
    for obj_key in objects_to_remove:
        del car_info[obj_key]

    # Display the processed frame
    cv2.imshow('Object detection', output_image)


def main(frame_skip=4):
    logging.info("Main function called...")  # Log: Main function called
    """Main function for the speed violation detection program."""

    # Define constants and parameters
    upper_roi_y, lower_roi_y = 320, 580
    line_distance_met = 40
    speed_limit_m = 15
    offset = 1

    # Open video file and retrieve frames per second
    video = cv2.VideoCapture("highway_vehicles.mp4")
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # Create output folder for speeding violation images
    output_folder = "violations"
    os.makedirs(output_folder, exist_ok=True)

    # Dictionary to store information about detected cars
    car_info = {}

    # Initialize frame count
    frame_count = 0

    # Main loop for processing video frames
    while True:
        # Read a frame from the video
        ret, frame = video.read()
        if not ret:
            break
        logging.debug("Before frame processing")    # Log: Before frame processing
        if frame_count % frame_skip == 0:
            logging.debug("Processing frame...")    # Log: Processing frame
            process_frame(frame, upper_roi_y, lower_roi_y, offset, car_info, line_distance_met, speed_limit_m, output_folder)
            logging.debug("After frame processing")   # Log: After frame processing

        # Increment frame count
        frame_count += 1

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) == ord('q'):
            break

        # Rewind video if it reaches the end
        if frame_count >= video.get(cv2.CAP_PROP_FRAME_COUNT):
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0


    # Release video capture and close windows
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    frame_skip = 8  # Adjust the frame_skip value as needed
    main(frame_skip)
