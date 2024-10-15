import cv2
import numpy as np
import cv2.aruco as aruco

print("OpenCV Version:", cv2.__version__)
image = cv2.imread("ar44100a.jpg")

if image is None:
    print("Cannot find the ArUco image!")
else:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define the ArUco dictionary (Change with the tag type)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        aruco.drawDetectedMarkers(image, corners, ids)

        intrinsic_matrix = np.array([[4015.98, 0, 2143.22], [0, 4026.50, 2181.04], [0, 0, 1]], dtype=np.float32)
        distortion_coeffs = np.array([0.21187366, -0.50634009, 0.00196516, 0.00467585, 0.21846391], dtype=np.float32)

        # The real size of the cube (m)
        tag_size = 0.1

        for i in range(len(ids)):

            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[i], tag_size, intrinsic_matrix, distortion_coeffs)

            # Calculate the central point
            tag_corners = corners[i].reshape((4, 2))  
            center_x = np.mean(tag_corners[:, 0])
            center_y = np.mean(tag_corners[:, 1])
            tag_center = (int(center_x), int(center_y))

            aruco.drawAxis(image, intrinsic_matrix, distortion_coeffs, rvec, tvec, 0.05)

            cube_points_3d = np.array([
                [-tag_size / 2, -tag_size / 2, 0],        
                [tag_size / 2, -tag_size / 2, 0],            
                [tag_size / 2, tag_size / 2, 0],          
                [-tag_size / 2, tag_size / 2, 0],           
                [-tag_size / 2, -tag_size / 2, tag_size],  
                [tag_size / 2, -tag_size / 2, tag_size],      
                [tag_size / 2, tag_size / 2, tag_size],        
                [-tag_size / 2, tag_size / 2, tag_size]        
            ], dtype=np.float32)

            cube_points_2d, _ = cv2.projectPoints(cube_points_3d, rvec, tvec, intrinsic_matrix, distortion_coeffs)
            cube_points_2d = np.int32(cube_points_2d).reshape(-1, 2)

            for point in cube_points_2d:
                cv2.circle(image, tuple(point), 5, (0, 255, 255), -1)

            # Draw the Cube
            image = cv2.drawContours(image, [cube_points_2d[:4]], -1, (0, 0, 255), 4)  # Red line, width: 4
            for j in range(4):
                image = cv2.line(image, tuple(cube_points_2d[j]), tuple(cube_points_2d[j + 4]), (0, 0, 255), 4)  
            image = cv2.drawContours(image, [cube_points_2d[4:]], -1, (0, 0, 255), 4)  

        output_filename = "output_ar44100a.jpg"
        cv2.imwrite(output_filename, image)
        print(f"File is saved as {output_filename}")

    else:
        print("Cannot detect the ArUco Pattern!")