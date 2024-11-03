import pdb  # Debugging
import glob  # File handling
import cv2  # OpenCV for image processing
import os  # OS functions
import numpy as np  # For array handling and matrix calculations

# Import custom functions or classes
from src.AshishRawat import some_function
from src.AshishRawat.some_folder import folder_func

# Detect and match features function
def detect_and_match_features(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    return src_pts, dst_pts

# Panorama stitching class
class PanaromaStitcher():
    def __init__(self):
        pass

    def make_panaroma_for_images_in(self, path):
        all_images = sorted(glob.glob(path + os.sep + '*'))
        print('Found {} images for stitching'.format(len(all_images)))
        self.say_hi()

        # Load images from the specified directory
        images = [cv2.imread(im) for im in all_images]

        # Stitch images and get the homographies
        final_panorama, homography_matrix_list = self.do_something_more(images)

        return final_panorama, homography_matrix_list

    def say_hi(self):
        print('Hi from Ashish Rawat..')

    def do_something_more(self, images):
        # Initialize the first image and a canvas for the panorama
        height, width = images[0].shape[:2]
        
        # Create a larger canvas to accommodate all images
        canvas_width = width * 3  # Total width for panorama
        canvas_height = height * 3  # Total height for panorama
        
        # Create an empty canvas
        panorama = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # Place the first image at the top-left corner (0, 0)
        panorama[0:height, 0:width] = images[0]

        accumulated_homography = np.eye(3)
        homography_matrix_list = []

        # Stitch each image in sequence
        for i in range(1, len(images)):
            # Calculate homography for the current pair of images
            H = self.do_something(images[i - 1], images[i])  # Homography for the current pair
            accumulated_homography = accumulated_homography @ H  # Accumulate transformations
            homography_matrix_list.append(accumulated_homography.copy())

            # Warp the current image according to the accumulated homography
            warped_image = cv2.warpPerspective(images[i], accumulated_homography, (canvas_width, canvas_height))
            mask = (warped_image > 0)  # Only overlay non-empty pixels
            panorama[mask] = warped_image[mask]

        return panorama, homography_matrix_list
    
    def do_something(self, img1, img2):
        # Detect and match features between two images
        src_pts, dst_pts = detect_and_match_features(img1, img2)
        # Calculate the homography matrix
        H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        return H
