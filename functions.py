import cv2
import numpy as np

def calibrate(image):
    height, width = image.shape[:2]
    hue_min, hue_max, saturation_min, saturation_max, value_min, value_max = 0,0,0,0,0,0
    warp_points = None 
    # Callback function for trackbars
    def nothing(x):
        pass

    # Create a window
    cv2.namedWindow('Trackbars')
    cv2.resizeWindow('Trackbars', 400, 400)

    cv2.createTrackbar('Separation 1', 'Trackbars', 0, int(image.shape[1] / 2), nothing)
    cv2.createTrackbar('Y Value 1', 'Trackbars', 0, int(image.shape[0] / 2), nothing)

    cv2.createTrackbar('Separation 2', 'Trackbars',  0, int(image.shape[1] / 2), nothing)
    cv2.createTrackbar('Y Value 2', 'Trackbars', 0, int(image.shape[0] / 2), nothing)

    cv2.createTrackbar('Hue Min', 'Trackbars', 0, 255, nothing)
    cv2.createTrackbar('Hue Max', 'Trackbars', 255, 255, nothing)
    cv2.createTrackbar('Saturation Min', 'Trackbars', 0, 255, nothing)
    cv2.createTrackbar('Saturation Max', 'Trackbars', 255, 255, nothing)
    cv2.createTrackbar('Value Min', 'Trackbars', 0, 255, nothing)
    cv2.createTrackbar('Value Max', 'Trackbars', 255, 255, nothing)

    while True: 
        separation1 = cv2.getTrackbarPos('Separation 1', 'Trackbars')
        y_value1 = cv2.getTrackbarPos('Y Value 1', 'Trackbars')

        separation2 = cv2.getTrackbarPos('Separation 2', 'Trackbars')
        y_value2 = cv2.getTrackbarPos('Y Value 2', 'Trackbars') + int(image.shape[0] / 2)

        hue_min = cv2.getTrackbarPos('Hue Min', 'Trackbars')
        hue_max = cv2.getTrackbarPos('Hue Max', 'Trackbars')
        saturation_min = cv2.getTrackbarPos('Saturation Min', 'Trackbars')
        saturation_max = cv2.getTrackbarPos('Saturation Max', 'Trackbars')
        value_min = cv2.getTrackbarPos('Value Min', 'Trackbars')
        value_max = cv2.getTrackbarPos('Value Max', 'Trackbars')
           
        # Draw circles at the warp points
        warp_point_img = image.copy()

        # Points for the first set of warp points
        warp_point1_x = separation1
        warp_point1_y = y_value1
        warp_point2_x = image.shape[1] - separation1
        warp_point2_y = y_value1

        # Points for the second set of warp points
        warp_point3_x = separation2
        warp_point3_y = y_value2
        warp_point4_x = image.shape[1] - separation2
        warp_point4_y = y_value2

        cv2.circle(warp_point_img, (warp_point1_x, warp_point1_y), 5, (0, 0, 255), -1)
        cv2.circle(warp_point_img, (warp_point2_x, warp_point2_y), 5, (0, 0, 255), -1)
        cv2.circle(warp_point_img, (warp_point3_x, warp_point3_y), 5, (255, 0, 0), -1)
        cv2.circle(warp_point_img, (warp_point4_x, warp_point4_y), 5, (255, 0, 0), -1)
 
        cv2.putText(warp_point_img, 'UL', (warp_point1_x + 15, warp_point1_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(warp_point_img, 'UR', (warp_point2_x - 30, warp_point2_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(warp_point_img, 'LL', (warp_point3_x + 15, warp_point3_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(warp_point_img, 'LR', (warp_point4_x - 30, warp_point4_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        
        warp_points = [[warp_point3_x, warp_point3_y],
                      [warp_point4_x, warp_point4_y],
                      [warp_point2_x, warp_point2_y],
                      [warp_point1_x, warp_point1_y]]
        unique_points_set = {tuple(point) for point in warp_points}
        unique_points_list = [list(point) for point in unique_points_set]
        if len(unique_points_list) == 4 :
            destination_points = [[0, height], [width, height], [width, 0], [0, 0]]
            target_points = np.float32(warp_points)
            destination = np.float32(destination_points)
            matrix = cv2.getPerspectiveTransform(target_points, destination)
            warped_result = cv2.warpPerspective(image, matrix, (width, height))  
            cv2.imshow('Warped Image', warped_result)
        else :
            warped_result = image
            cv2.imshow('Warped Image',image)
            
                # Convert the image to HSV
        hsv = cv2.cvtColor(warped_result, cv2.COLOR_BGR2HSV)

        # Create a mask based on the HSV thresholds
        hsv_min = np.array([hue_min, saturation_min, value_min])
        hsv_max = np.array([hue_max, saturation_max, value_max])
        mask = cv2.inRange(hsv, hsv_min, hsv_max)
                # Display the images
        cv2.imshow('Original Image', image)
        cv2.imshow('HSV Threshold Image', mask)
        cv2.imshow('Warp Points Image', warp_point_img)
        
        # Check for the 'Esc' key press
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            break

    # Destroy the window and return the trackbar values and warp points
    return [ [hue_min,  saturation_min ,value_min ], [hue_max , saturation_max, value_max] , warp_points,mask]

def draw_steering_indicator(image, steering_value, radius = 50,text_offset = [10,-5],text_size = 0.8):
    angle_rad = np.interp(steering_value, [-1, 0, 1], [np.pi, np.pi/2, 0])
    center_x, center_y = image.shape[1] // 2, int(image.shape[0] * 0.9)
    end_x = int(center_x + radius * np.cos(angle_rad))
    end_y = int(center_y - radius * np.sin(angle_rad))
    cv2.ellipse(image, (center_x, center_y), (radius, radius), 180, 180, 0, 255, 3)
    cv2.line(image, (center_x, center_y), (end_x, end_y), (0, 255, 0), 3)
    text_offset_x, text_offset_y = text_offset[0], text_offset[1]
    text_x = end_x + text_offset_x
    text_y = end_y + text_offset_y
    text = f'Steering: {steering_value:.2f}'
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 255), 2)
    cv2.circle(image, (end_x, end_y), radius//10, (60, 200, 0), -1)
    cv2.circle(image, (center_x, center_y), radius//15, (255, 255, 255), -1)
    return image

def create_combined_frame(images):
    # Resize each image to 400x400
    resized_images = [cv2.resize(image, (300, 300)) for image in images]

    # Create a blank image with size 800x800 (2x2 grid)
    combined_frame = np.zeros((600, 600, 3), dtype=np.uint8)

    # Paste each resized image into the combined frame
    for i in range(2):
        for j in range(2):
            combined_frame[i * 300:(i + 1) * 300, j * 300:(j + 1) * 300, :] = resized_images[i * 2 + j]

    return combined_frame

def sliding_window(mask):
    #Histogram
    histogram = np.sum(mask[mask.shape[0]//2 : , : ],axis = 0) #sum the pixel value of the masked image from half image to bottom
    #now sepearate left lane and right lane
    midpoint = np.int(histogram.shape[0]/2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    #sliding windows
    y = mask.shape[0] - 10
    number_of_sliding_windows = 12
    slid_h  = int ( y  / number_of_sliding_windows ) # sliding_window_height
    slid_w = 80 # sliding_window_width
    left_points = []
    right_points = []
    lx = []
    rx = []
    msk = mask.copy()
    msk = cv2.merge([msk, msk, msk]) 
    while y > 0 :
        ## left threshold
        temp_img = mask[y- slid_h : y , left_base - int(slid_w/2) : left_base + int(slid_w/2)]
        contours, _ = cv2.findContours(temp_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours :
            M = cv2.moments(contour)
            if M["m00"] != 0 :
                cx = int(M["m10"]/M["m00"])  #find center of mass of the contour, x value in that window box
                cy = int(M["m01"]/M["m00"])
                lx.append(left_base - int(slid_w/2) + cx)
                left_base = left_base - int(slid_w/2) + cx # New left base for next window

        ## Right threshold
        temp_img = mask[y- slid_h : y , right_base - int(slid_w/2) : right_base + int(slid_w/2)]
        contours, _ = cv2.findContours(temp_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours :
            M = cv2.moments(contour)
            if M["m00"] != 0 :
                cx = int(M["m10"]/M["m00"])  #find center of mass of the contour, x value in that window box
                cy = int(M["m01"]/M["m00"])
                lx.append(right_base - int(slid_w/2) + cx)
                right_base = right_base - int(slid_w/2) + cx # New left base for next window

        left_points.append((left_base, y - int(slid_h/2)))
        right_points.append((right_base, y - int(slid_h/2)) )
        cv2.circle(msk, (left_base, y - int(slid_h/2)) , 2, (0, 0, 255), 2)
        cv2.circle(msk, (right_base, y - int(slid_h/2)) , 2, (0, 0, 255), 2)
        cv2.rectangle(msk,  (left_base - int(slid_w/2),y),(left_base + int(slid_w/2),y - slid_h) , (255,255,0) , 2 )
        cv2.rectangle(msk,  (right_base - int(slid_w/2),y),(right_base + int(slid_w/2),y - slid_h) , (255,255,0) , 2 )
        y -= slid_h
    return left_points,right_points,msk


def map_to_range(value, from_min, from_max, to_min, to_max):
    # Map a value from one range to another
    return (value - from_min) * (to_max - to_min) / (from_max - from_min) + to_min


def find_steering_angle(left_lane_points, right_lane_points, image_width):
    # Convert the lists of tuple coordinates to numpy arrays
    left_lane_points = np.array(left_lane_points)
    right_lane_points = np.array(right_lane_points)

    # Fit a second-degree polynomial to the left and right lane points
    left_fit = np.polyfit(left_lane_points[:, 1], left_lane_points[:, 0], 2)
    right_fit = np.polyfit(right_lane_points[:, 1], right_lane_points[:, 0], 2)

    # Calculate the curvature of the lanes
    y_eval = image_width  # Evaluate curvature at the bottom of the image
    left_curvature = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.abs(2 * left_fit[0])
    right_curvature = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.abs(2 * right_fit[0])

    # Calculate the offset of the vehicle from the center of the lanes
    lane_center = (left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2] +
                   right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]) / 2
    vehicle_center = image_width / 2
    offset = lane_center - vehicle_center

    # Assume the camera is mounted at the center of the vehicle and the focal length is 1
    focal_length = 10

    # Calculate the steering angle based on the offset
    steering_angle = np.arctan(offset / focal_length) * (180 / np.pi)
    steering_angle = map_to_range(steering_angle,-90,90,-1,1)
    return steering_angle


