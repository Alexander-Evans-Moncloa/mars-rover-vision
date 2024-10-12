import cv2
import numpy as np
import socket
import paho.mqtt.client as mqtt

# ArUco and MQTT setup (same as before)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
corner_ids = [0, 16, 28, 29]
drone_id = 30
real_world_coords = {0: (0.0, 0.0), 16: (0.5, 0.0), 28: (0.0, 0.5), 29: (0.5, 0.5)}
ARUCO_MARKER_SIZE_M = 0.10
client = mqtt.Client()
client.username_pw_set("CENSORED", password="CENSORED")
client.connect("CENSORED", 31415, 60)
client.loop_start()

# MQTT Topics for alien points (red objects)
MQTT_TOPIC_P1_X = 'project_sisyphus/point_1_x'
MQTT_TOPIC_P1_Y = 'project_sisyphus/point_1_y'
MQTT_TOPIC_P2_X = 'project_sisyphus/point_2_x'
MQTT_TOPIC_P2_Y = 'project_sisyphus/point_2_y'
MQTT_TOPIC_P3_X = 'project_sisyphus/point_3_x'
MQTT_TOPIC_P3_Y = 'project_sisyphus/point_3_y'
MQTT_TOPIC_P4_X = 'project_sisyphus/point_4_x'
MQTT_TOPIC_P4_Y = 'project_sisyphus/point_4_y'
MQTT_TOPIC_P5_X = 'project_sisyphus/point_5_x'
MQTT_TOPIC_P5_Y = 'project_sisyphus/point_5_y'

# Other MQTT Topics (same as before)
MQTT_TOPIC_X = 'project_sisyphus/current_ball_position_x'           # ArUco code x coord
MQTT_TOPIC_Y = 'project_sisyphus/current_ball_position_y'           # ArUco code y coord
MQTT_TOPIC_YAW = 'project_sisyphus/yaw_angle'                       # ArUco code 30 angle
MQTT_TOPIC_EVENT_STATE = 'project_sisyphus/critical_event_state'    # State is 0 when no green ball detected, 1 when detected
MQTT_TOPIC_EVENT_X = 'project_sisyphus/critical_event_x'            # Coordinates of green ball
MQTT_TOPIC_EVENT_Y = 'project_sisyphus/critical_event_y'            # Coordinates of green ball
MQTT_TOPIC_EVENT_THETA = 'project_sisyphus/critical_event_theta'    # Green ball angle

# MQTT Topics to add to main
MQTT_TOPIC_STATE = 'project_sisyphus/program_state'  # 0 for idle, 1 for setup, 2 for start
MQTT_TOPIC_X_ARENA = 'project_sisyphus/arena_x'      # to be 500 automatically during setup phase
MQTT_TOPIC_Y_ARENA = 'project_sisyphus/arena_y'      # to be 500 automatically during setup phase

# Helper functions (same as before)
def detect_aruco_markers(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    return corners, ids

def estimate_homography(corner_coords):
    if len(corner_coords) < 4:
        return None
    real_points = []
    image_points = []
    for marker_id, (cx, cy) in corner_coords.items():
        real_points.append(real_world_coords[marker_id])
        image_points.append([cx, cy])
    real_points = np.array(real_points, dtype="float32")
    image_points = np.array(image_points, dtype="float32")
    H, _ = cv2.findHomography(image_points, real_points)
    return H

def apply_homography(H, image_point):
    """Convert image coordinates to real-world coordinates using homography."""
    image_point = np.array([image_point[0], image_point[1], 1]).reshape((3, 1))
    world_point = np.dot(H, image_point)
    world_point /= world_point[2]  # Normalize the point by dividing by the Z coordinate
    return world_point[0], world_point[1]

def calculate_yaw_angle(corners0, corners_drone):
    """Calculate the yaw angle of ArUco marker 30 relative to ArUco marker 0."""
    vector_0 = corners0[1] - corners0[0]  # Orientation vector of marker 0
    vector_drone = corners_drone[1] - corners_drone[0]  # Orientation vector of drone marker

    angle_0 = np.degrees(np.arctan2(vector_0[1], vector_0[0])) % 360
    angle_drone = np.degrees(np.arctan2(vector_drone[1], vector_drone[0])) % 360

    yaw_angle = (angle_drone - angle_0) % 360
    return yaw_angle

# Red object detection function
def detect_red_objects(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv_frame, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    red_points = []
    for contour in contours:
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        if radius > 5:
            red_points.append((int(x), int(y)))
            if len(red_points) == 5:  # Limit to 5 points
                break
    return red_points

def publish_alien_objects(red_points):
    mqtt_topics_x = [MQTT_TOPIC_P1_X, MQTT_TOPIC_P2_X, MQTT_TOPIC_P3_X, MQTT_TOPIC_P4_X, MQTT_TOPIC_P5_X]
    mqtt_topics_y = [MQTT_TOPIC_P1_Y, MQTT_TOPIC_P2_Y, MQTT_TOPIC_P3_Y, MQTT_TOPIC_P4_Y, MQTT_TOPIC_P5_Y]
    
    for i, (x, y) in enumerate(red_points):
        client.publish(mqtt_topics_x[i], f"{float(x):.2f}")
        client.publish(mqtt_topics_y[i], f"{float(y):.2f}")

def detect_green_ball(frame):
    """Detect the green ball in the frame and return its coordinates."""
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        if radius > 5:
            return int(x), int(y)
    return None

def calculate_ball_angle(previous_position, current_position, marker0_angle):
    """Calculate ball movement angle relative to ArUco marker 0's orientation."""
    dx = current_position[0] - previous_position[0]
    dy = current_position[1] - previous_position[1]
    ball_angle = np.degrees(np.arctan2(dy, dx)) % 360

    # Adjust ball angle relative to ArUco marker 0's orientation
    relative_ball_angle = (ball_angle - marker0_angle) % 360
    return relative_ball_angle

def main():
    # Set up video capture
    cap = cv2.VideoCapture(0)
    last_valid_homography = None
    previous_ball_center = None
    program_state = 0  # 0 = idle, 1 = setup, 2 = start
    client.publish(MQTT_TOPIC_STATE, str(program_state))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ArUco marker detection
        corners, ids = detect_aruco_markers(frame)

        corner_coords = {}
        if ids is not None:
            ids = ids.flatten()
            for i, marker_id in enumerate(ids):
                if marker_id in corner_ids:
                    c = corners[i][0]
                    marker_center = (np.mean(c[:, 0]), np.mean(c[:, 1]))
                    corner_coords[marker_id] = marker_center

        # Homography estimation
        H = estimate_homography(corner_coords)

        if H is not None:
            last_valid_homography = H
        elif last_valid_homography is not None:
            H = last_valid_homography

        # ArUco marker and drone detection, along with yaw calculation
        if H is not None and ids is not None and drone_id in ids and 0 in ids:
            program_state = 2  # Start program
            client.publish(MQTT_TOPIC_STATE, str(program_state))

            drone_index = np.where(ids == drone_id)[0][0]
            marker0_index = np.where(ids == 0)[0][0]
            c_drone = corners[drone_index][0]
            c_marker0 = corners[marker0_index][0]

            # Drone world coordinates and yaw
            drone_center = (np.mean(c_drone[:, 0]), np.mean(c_drone[:, 1]))
            drone_world_coords = apply_homography(H, drone_center)
            x_coord = float(drone_world_coords[0])
            y_coord = float(drone_world_coords[1])
            yaw_angle = float(calculate_yaw_angle(c_marker0, c_drone))

            # Publish drone coordinates and yaw to MQTT
            client.publish(MQTT_TOPIC_X, f"{x_coord:.2f}")
            client.publish(MQTT_TOPIC_Y, f"{y_coord:.2f}")
            client.publish(MQTT_TOPIC_YAW, f"{yaw_angle:.2f}")

            # Detect green ball and transform its coordinates
            ball_center = detect_green_ball(frame)
            if ball_center:
                ball_world_coords = apply_homography(H, ball_center)
                ball_x = float(ball_world_coords[0])
                ball_y = float(ball_world_coords[1])

                # Draw a green dot on the detected green ball location
                cv2.circle(frame, ball_center, 5, (0, 255, 0), -1)

                if previous_ball_center:
                    previous_ball_world_coords = apply_homography(H, previous_ball_center)
                    marker0_vector = c_marker0[1] - c_marker0[0]
                    marker0_angle = np.degrees(np.arctan2(marker0_vector[1], marker0_vector[0])) % 360
                    ball_angle = float(calculate_ball_angle(previous_ball_world_coords, ball_world_coords, marker0_angle))

                    # Publish green ball information
                    client.publish(MQTT_TOPIC_EVENT_STATE, "1")
                    client.publish(MQTT_TOPIC_EVENT_X, f"{ball_x:.2f}")
                    client.publish(MQTT_TOPIC_EVENT_Y, f"{ball_y:.2f}")
                    client.publish(MQTT_TOPIC_EVENT_THETA, f"{ball_angle:.2f}")
                previous_ball_center = ball_center
            else:
                client.publish(MQTT_TOPIC_EVENT_STATE, "0")

            # Detect and transform red objects
            red_points = detect_red_objects(frame)
            red_world_coords = [apply_homography(H, point) for point in red_points]

            # Publish transformed red points to MQTT and draw red points with indices
            mqtt_topics_x = [MQTT_TOPIC_P1_X, MQTT_TOPIC_P2_X, MQTT_TOPIC_P3_X, MQTT_TOPIC_P4_X, MQTT_TOPIC_P5_X]
            mqtt_topics_y = [MQTT_TOPIC_P1_Y, MQTT_TOPIC_P2_Y, MQTT_TOPIC_P3_Y, MQTT_TOPIC_P4_Y, MQTT_TOPIC_P5_Y]
            
            for i, (x_world, y_world) in enumerate(red_world_coords):
                x_world_float = float(x_world)
                y_world_float = float(y_world)
                client.publish(mqtt_topics_x[i], f"{x_world_float:.2f}")
                client.publish(mqtt_topics_y[i], f"{y_world_float:.2f}")
                # Draw the points in world coordinates on the frame
                cv2.putText(frame, str(i + 1), red_points[i], cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Draw detected markers and additional information on the frame
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Display basic MQTT information on frame
        if H is not None:
            cv2.putText(frame, f"Drone: X={x_coord:.2f}, Y={y_coord:.2f}, Yaw={yaw_angle:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if ball_center:
                cv2.putText(frame, f"Ball: X={ball_x:.2f}, Y={ball_y:.2f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            for i, (x_world, y_world) in enumerate(red_world_coords):
                x_world_float = float(x_world[0])
                y_world_float = float(y_world[0])
                cv2.putText(frame, f"Red {i + 1}: X={x_world_float:.2f}, Y={y_world_float:.2f}",
                            (10, 90 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show the frame
        cv2.imshow('Frame', frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
