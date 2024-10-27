import cv2
import numpy as np
import socket
import paho.mqtt.client as mqtt
import heapq
import itertools
import time

# ArUco and MQTT setup (same as before)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
corner_ids = [0, 16, 28, 29]
drone_id = 30
real_world_coords = {0: (0.0, 0.0), 16: (1000, 0.0), 28: (0.0, 1000), 29: (1000, 1000)}
ARUCO_MARKER_SIZE_M = 100
client = mqtt.Client()
client.username_pw_set("CENSORED", password="CENSORED")
client.connect("CENSORED", 31415, 60)
client.loop_start()

# The next position for rover to go towards
MQTT_TOPIC_TARGET_X = 'project_sisyphus/target_position_x'
MQTT_TOPIC_TARGET_Y = 'project_sisyphus/target_position_y'

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
    """Calculate the yaw angle of ArUco marker 30 relative to ArUco marker 0 in radians."""
    vector_0 = corners0[1] - corners0[0]  # Orientation vector of marker 0
    vector_drone = corners_drone[1] - corners_drone[0]  # Orientation vector of drone marker

    # Calculate the orientation angles in radians
    angle_0 = np.arctan2(vector_0[1], vector_0[0]) % (2 * np.pi)
    angle_drone = np.arctan2(vector_drone[1], vector_drone[0]) % (2 * np.pi)

    # Calculate the yaw angle difference in radians
    yaw_angle = (angle_drone - angle_0) % (2 * np.pi)
    
    return yaw_angle

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_obstacle_near_line(p1, p2, obstacle, threshold):
    """Check if an obstacle is within a certain distance from the line segment between p1 and p2."""
    p1, p2 = np.array(p1), np.array(p2)
    obstacle = np.array(obstacle)

    # Calculate the distance from the obstacle to the line segment
    line_vec = p2 - p1
    p1_to_obstacle = obstacle - p1
    line_len = np.linalg.norm(line_vec)
    
    # Check if line is a point
    if line_len == 0:
        return np.linalg.norm(p1_to_obstacle) < threshold
    
    line_unit_vec = line_vec / line_len
    projection_length = np.dot(p1_to_obstacle, line_unit_vec)

    # Find the closest point on the line segment to the obstacle
    if projection_length < 0:
        closest_point = p1
    elif projection_length > line_len:
        closest_point = p2
    else:
        closest_point = p1 + projection_length * line_unit_vec
    
    # Distance from the obstacle to the closest point on the line
    distance_to_line = np.linalg.norm(obstacle - closest_point)
    
    return distance_to_line < threshold

def detect_red_objects(H, frame, x_coord, y_coord, num_points=6):
    """Detect red objects and return points sorted by distance to (x_coord, y_coord)."""
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
            if len(red_points) == num_points:  # Limit to num_points
                break
    if len(red_points) < 2:
        return None, None, []  # Not enough points detected
    
    # Sort points by vector distance to the given x_coord and y_coord
    red_points.sort(key=lambda point: np.linalg.norm(np.array(apply_homography(H, [point[0], point[1]])) - np.array([x_coord, y_coord])))
    
    point_1 = red_points[0]
    point_last = red_points[-1]
    middle_points = red_points[1:-1]
    num_points = num_points
    
    return point_1, point_last, middle_points, num_points

def detect_blue_obstacles(frame, max_obstacles=2):
    """Detect blue obstacles in the frame."""
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    blue_obstacles = []
    for contour in contours:
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        if radius > 5:
            blue_obstacles.append((int(x), int(y)))
            if len(blue_obstacles) == max_obstacles:
                break
    return blue_obstacles

def tsp_with_obstacle_avoidance(start_point, end_point, middle_points, blue_obstacles, obstacle_threshold=10):
    """Solve TSP while avoiding blue obstacles by expanding their size."""
    optimal_order = None
    min_total_distance = float('inf')
    
    # Test all permutations of middle points
    for perm in itertools.permutations(middle_points):
        p_list = [start_point] + list(perm) + [end_point]
        obstacle_in_path = False
        
        # Check for obstacles between consecutive points
        for i in range(len(p_list) - 1):
            for obstacle in blue_obstacles:
                if is_obstacle_near_line(p_list[i], p_list[i + 1], obstacle, obstacle_threshold):
                    obstacle_in_path = True
                    break
            if obstacle_in_path:
                break
        
        if obstacle_in_path:
            continue
        
        # Calculate total distance for this permutation
        total_distance = sum(calculate_distance(p_list[i], p_list[i + 1]) for i in range(len(p_list) - 1))
        
        if total_distance < min_total_distance:
            min_total_distance = total_distance
            optimal_order = perm
    
    # Return the optimal order
    return [start_point] + list(optimal_order) + [end_point] if optimal_order else [start_point] + middle_points + [end_point]

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
    """Calculate ball movement angle relative to ArUco marker 0's orientation in radians."""
    dx = current_position[0] - previous_position[0]
    dy = current_position[1] - previous_position[1]
    
    # Calculate the angle of the ball's movement in radians
    ball_angle = np.arctan2(dy, dx) % (2 * np.pi)

    # Adjust ball angle relative to ArUco marker 0's orientation, assuming marker0_angle is in radians
    relative_ball_angle = (ball_angle - marker0_angle) % (2 * np.pi)
    
    return relative_ball_angle

def target_coordinate_calculation(x_coord, y_coord, current_index, red_world_coords, endpoint_coordinate_x, endpoint_coordinate_y):
    """
    Determine the next target coordinates based on the red object positions.
    
    Args:
    - x_coord (float): The x-coordinate of the drone.
    - y_coord (float): The y-coordinate of the drone.
    - current_index (int): The current index for the target coordinate arrays.
    - red_world_coords (list): List of red object world coordinates.
    - endpoint_coordinate_x (float): The x-coordinate of the final endpoint.
    - endpoint_coordinate_y (float): The y-coordinate of the final endpoint.
    
    Returns:
    - target_coordinate_x (float): The next target x-coordinate.
    - target_coordinate_y (float): The next target y-coordinate.
    - current_index (int): Updated index for the target coordinate arrays.
    - finished (bool): Whether the final target has been reached.
    """
    # Extract the x and y coordinates of the red objects
    target_coordinate_x_array = [point[0].item() for point in red_world_coords]
    target_coordinate_y_array = [point[1].item() for point in red_world_coords]

    # If no red objects are detected, return the current coordinates
    if not target_coordinate_x_array or not target_coordinate_y_array:
        return x_coord, y_coord, current_index, False

    # Initialize finished as False
    finished = False

    # Initialize target coordinates
    target_coordinate_x = endpoint_coordinate_x
    target_coordinate_y = endpoint_coordinate_y

    # If we haven't reached the last red marker yet
    if current_index < len(target_coordinate_x_array):
        target_coordinate_x = target_coordinate_x_array[current_index]
        target_coordinate_y = target_coordinate_y_array[current_index]
        
        # Check if the drone is within 50 units of the current red object
        if abs(x_coord - target_coordinate_x) <= 50 and abs(y_coord - target_coordinate_y) <= 50:
            # Move to the next red object or the endpoint if on the last red marker
            if current_index == len(target_coordinate_x_array) - 1:
                current_index = 7  # Lock index at 7 once at the endpoint
            else:
                current_index += 1

    # Check if the drone has reached the endpoint
    if current_index == 7 and abs(x_coord - endpoint_coordinate_x) <= 50 and abs(y_coord - endpoint_coordinate_y) <= 50:
        finished = True
        print("JOURNEY FINISHED!")

    # Return the next target coordinates
    return target_coordinate_x, target_coordinate_y, current_index, finished

def main():
    # Set up video capture
    cap = cv2.VideoCapture(0)
    last_valid_homography = None
    program_state = 0  # 0 = idle, 1 = start, 2 = finish
    current_index = 0
    initialization_complete = False  # Flag to indicate initialization is complete

    client.publish(MQTT_TOPIC_STATE, str(program_state))

    previous_position = None  # Initialize previous position as None

    endpoint_coordinate_x = float(input("Enter the new final x-coordinate: "))
    endpoint_coordinate_y = float(input("Enter the new final y-coordinate: "))

    # Initialize a timer
    start_time = time.time()
    elapsed_time = 0
    waiting_for_input = True  # Flag to control when to wait for input

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
                        
            client.publish(MQTT_TOPIC_STATE, str(program_state))

            drone_index = np.where(ids == drone_id)[0][0]
            marker0_index = np.where(ids == 0)[0][0]
            c_drone = corners[drone_index][0]
            c_marker0 = corners[marker0_index][0]

            # Gets endpoint coordinates relative to real coordinates then shows them with white circle
            raw_coordinates = (endpoint_coordinate_x, endpoint_coordinate_y)
            endpoint_coords = apply_homography(np.linalg.inv(H), raw_coordinates)
            endpoint_coordinate_x_real = endpoint_coords[0].item()
            endpoint_coordinate_y_real = endpoint_coords[1].item()
            cv2.circle(frame, (int(endpoint_coordinate_x_real), int(endpoint_coordinate_y_real)), 20, (255, 255, 255), 5)

            # Drone world coordinates and yaw
            drone_center = (np.mean(c_drone[:, 0]), np.mean(c_drone[:, 1]))
            drone_world_coords = apply_homography(H, drone_center)
            x_coord = int(drone_world_coords[0].item())  # Fix deprecated conversion
            y_coord = int(drone_world_coords[1].item())  # Fix deprecated conversion
            yaw_angle = round(float(calculate_yaw_angle(c_marker0, c_drone)), 1)  # This is your marker0_angle

            # Check if initialization is complete
            if not initialization_complete:
                # Detect red obstacles and blue obstacles
                point_1, point_last, middle_points, num_points = detect_red_objects(H, frame, x_coord, y_coord)
                blue_obstacles = detect_blue_obstacles(frame)

                # Ensure we have enough red points
                if point_1 and point_last and len(middle_points) == (num_points - 2):
                    # Apply TSP with obstacle avoidance
                    ordered_points = tsp_with_obstacle_avoidance(point_1, point_last, middle_points, blue_obstacles)
                    # Transform red points to world coordinates using homography
                    red_world_coords = [apply_homography(H, point) for point in ordered_points]

                    # Set the flag to indicate initialization is complete
                    initialization_complete = True
                else:
                    print("Not enough red points detected.")
                    continue  # Skip the rest of the loop if we don't have enough points

            # Get target coordinates from the function
            target_coord_x, target_coord_y, current_index, finished = target_coordinate_calculation(x_coord, y_coord, current_index, red_world_coords, endpoint_coordinate_x, endpoint_coordinate_y)

            if program_state == 1:  # Ensure the navigation starts after the user input
                # Continue with navigation logic...
                if finished:  # Only transition to state 2 after navigation is finished
                    program_state = 2
                    client.publish(MQTT_TOPIC_STATE, str(program_state))

            target_coord_x = int(target_coord_x)
            target_coord_y = int(target_coord_y)

            # Define the real-world coordinates
            real_world_coords = (target_coord_x, target_coord_y)

            # Apply homography to transform real-world coordinates to image coordinates
            image_coords = apply_homography(np.linalg.inv(H), real_world_coords)  # Invert H to go from real-world to image

            # Convert image_coords to integers (OpenCV expects integers for the center point)
            image_coords_int = (int(image_coords[0]), int(image_coords[1]))

            # Draw a circle at the corresponding image coordinate
            cv2.circle(frame, image_coords_int, 50, (225, 0, 0), 1)

            # Publish drone coordinates and yaw to MQTT
            client.publish(MQTT_TOPIC_X, f"{x_coord:.2f}")
            client.publish(MQTT_TOPIC_Y, f"{y_coord:.2f}")
            client.publish(MQTT_TOPIC_YAW, f"{yaw_angle:.2f}")

            # Publish target coordinates to MQTT
            client.publish(MQTT_TOPIC_TARGET_X, f"{target_coord_x:.2f}")
            client.publish(MQTT_TOPIC_TARGET_Y, f"{target_coord_y:.2f}")

            # Detect green ball and calculate angle
            ball_center = detect_green_ball(frame)

            if ball_center:
                ball_world_coords = apply_homography(H, ball_center)
                ball_x = int(ball_world_coords[0].item())  # Fix deprecated conversion
                ball_y = int(ball_world_coords[1].item())  # Fix deprecated conversion

                # Current position of the ball (newly detected)
                current_position = (ball_x, ball_y)

                # If this is the first detection, set previous_position equal to current_position
                if previous_position is None:
                    previous_position = current_position

                # Calculate the ball angle relative to marker 0's orientation (yaw_angle)
                relative_ball_angle = calculate_ball_angle(previous_position, current_position, yaw_angle)

                relative_ball_angle = round(relative_ball_angle, 1)

                # Update previous_position for the next frame
                previous_position = current_position

                # Draw a circle on the ball in the image
                cv2.circle(frame, ball_center, 5, (0, 255, 0), -1)

                # Publish green ball information
                client.publish(MQTT_TOPIC_EVENT_STATE, "1")
                client.publish(MQTT_TOPIC_EVENT_X, f"{ball_x:.2f}")
                client.publish(MQTT_TOPIC_EVENT_Y, f"{ball_y:.2f}")
                client.publish(MQTT_TOPIC_EVENT_THETA, f"{relative_ball_angle:.2f}")
            else:
                client.publish(MQTT_TOPIC_EVENT_STATE, "0")

            # Draw red object coordinates without blue circles
            for i, (x_world, y_world) in enumerate(red_world_coords):
                x_world_float = x_world[0]
                y_world_float = y_world[0]
                red_image_coords = apply_homography(np.linalg.inv(H), (x_world_float, y_world_float))
                cv2.circle(frame, (int(red_image_coords[0]), int(red_image_coords[1])), 20, (0, 0, 255), 5)
                cv2.putText(frame, str(i + 1), ordered_points[i], cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)

            # Draw blue obstacles with blue circles
            for x, y in blue_obstacles:
                cv2.circle(frame, (x, y), 20, (255, 0, 0), 5)  # Blue circles around blue obstacles

            # Publish program state to MQTT
            client.publish(MQTT_TOPIC_STATE, str(program_state))

        # Draw detected markers and additional information on the frame
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        if 30 in ids:
            marker_detection_start_time = None  # Reset timer if all markers are detected
        else:
            # Start the timer if it wasn't already started
            if marker_detection_start_time is None:
                marker_detection_start_time = time.time()

            # If markers have not been detected for 2 seconds, change to program_state = 2
            elif time.time() - marker_detection_start_time >= 2:
                program_state = 2
                client.publish(MQTT_TOPIC_STATE, str(program_state))

        # Display basic MQTT information on frame
        if H is not None:
            cv2.putText(frame, f"Drone: X={x_coord:.2f}, Y={y_coord:.2f}, Yaw={yaw_angle:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if ball_center:
                cv2.putText(frame, f"Ball: X={ball_x:.2f}, Y={ball_y:.2f}, Theta={relative_ball_angle:.2f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            x_world_float = int(real_world_coords[0])  # Fix deprecated conversion
            y_world_float = int(real_world_coords[1])  # Fix deprecated conversion
            
            if not finished:
                cv2.putText(frame, f"Target: X={x_world_float:.0f}, Y={y_world_float:.0f}",
                            (10, 90 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                cv2.putText(frame, f"TARGET REACHED",
                            (10, 90 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow('Drone Navigation', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Check if 1 second has passed
        elapsed_time = time.time() - start_time
        
        if elapsed_time >= 1 and waiting_for_input:
            print("Press Enter to start the program after confirming the points are correct...")
            user_input = input()  # Wait for user input
            
            if user_input.lower() == 'q':
                break
            else:
                program_state = 1  # Change to start state after input
                client.publish(MQTT_TOPIC_STATE, str(program_state))  # Publish the updated state
                waiting_for_input = False  # Stop waiting for input


    cap.release()
    cv2.destroyAllWindows()    

if __name__ == "__main__":
    main()
