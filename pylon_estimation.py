import cv2
import numpy as np

image = cv2.imread('pylon.jpg')
size = image.shape

# The coordinate of the points on the 2D image plane
image_points = np.array(   [(1323.51, 2343.51),     # Bottom left edge
                            (1946.11, 2389.88),     # Bottom right edge
                            (1293.71, 883.058),     # Top left edge
                            (2111.69, 585.006),     # Top right edge
                            (1230.79, 614.812),     # Middle left edge
                            (2115.01, 591.63)      # Middle right edge
                        ], dtype="double")

# The coordinate of the points on the 3D world model
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Bottom left edge
                            (12.4, 0.0, 0.0),            # Bottom right edge
                            (-1.8, 28.3, 4.7),     # Top left edge
                            (14.2, 28.2, 5.8),      # Top right edge
                            (-2.7, 35.3, 5.6),    # Middle left edge
                            (16.1, 34.1, 5.8)      # Middle right edge
                         
                        ])


camera_matrix = np.array([[632.92190286, 0, 253.29077481],
				   			[0, 630.10778062, 318.92409896],
				   			[0,0,1]], dtype="double")

print ("Camera matrix is \n: {0} " .format(camera_matrix))

diff_coeff = np.array([ 0.17165397, -0.34693743, -0.00293247,  0.00966238,  0.57375437])
(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, diff_coeff)

print ("Rotation vector is \n: {0} " .format(rotation_vector))
print ("Translation vector is \n: {0} " .format(translation_vector))

(end_points, jacobian) = cv2.projectPoints(np.array([(0.0,0.0,1000.0)]),rotation_vector,translation_vector,camera_matrix,diff_coeff)
for p in image_points:
 	cv2.circle(image, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

p1 = (int(image_points[0][0]), int(image_points[0][1]))
p2 = (int(end_points[0][0][0]), int(end_points[0][0][1]))

cv2.line(image, p1, p2, (255,0,0), 2)

reduced_image = cv2.resize(image, (0,0), fx = 0.2, fy = 0.2)
cv2.imshow("Output", reduced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
