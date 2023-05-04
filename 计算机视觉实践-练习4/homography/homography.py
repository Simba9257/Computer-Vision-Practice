import cv2
import skimage
from skimage import img_as_ubyte
import numpy as np


def matchPics(I1, I2):
    # Given two images I1 and I2, perform SIFT matching to find candidate match pairs

    ### YOUR CODE HERE
    ### You can use skimage or OpenCV to perform SIFT matching

    # Initializing empty lists
    locs1 = []
    locs2 = []
    matches = []

    # Convert images to 8 bits
    I1_converted = img_as_ubyte(I1)
    I2_converted = img_as_ubyte(I2)

    # Initialize ORB and matcher objects
    orb = cv2.ORB_create()
    matcher = cv2.BFMatcher()

    # Compute descriptors and keypoints
    keypoint1, descriptor1 = orb.detectAndCompute(I1_converted, None)
    keypoint2, descriptor2 = orb.detectAndCompute(I2_converted, None)

    # Matches descriptors of keypoints
    matches = matcher.match(descriptor1, descriptor2)

    # Sort points by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Store coordinates of keypoints for each match
    locs1 = [keypoint1[m.queryIdx].pt for m in matches]
    locs2 = [keypoint2[m.trainIdx].pt for m in matches]

    # Convert in array
    matches = np.array([[m.queryIdx, m.trainIdx] for m in matches])
    ### END YOUR CODE

    return matches, locs1, locs2

def computeH_ransac(matches, locs1, locs2):
    iterations = 100000
    threshold = 5

    locs1 = np.array(locs1)
    locs2 = np.array(locs2)

    max_inliers = np.zeros(matches.shape[0])
    bestH = None
    inliers = None

    # RANSAC loop
    for i in range(iterations):
        # Randomly select four matches
        indices = np.random.choice(matches.shape[0], 4, replace=False)
        sample_locs1 = locs1[indices]
        sample_locs2 = locs2[indices]

        # Compute homography using the four matches
        H = computeH(sample_locs1, sample_locs2)

        # Apply homography to all matches
        warped_locs1 = applyH(H, locs1)
        d = np.linalg.norm(warped_locs1 - locs2, axis=1)

        # Count inliers
        inliers = d < threshold
        n_inliers = np.sum(inliers)

        # Check if this is the best estimate so far
        if n_inliers > np.sum(max_inliers):
            max_inliers = inliers
            bestH = H
            print(bestH)

    return bestH, max_inliers.astype(bool)

def computeH(locs1, locs2):
    n = locs1.shape[0]
    A = np.zeros((2 * n, 9))
    for i in range(n):
        x1, y1 = locs1[i]
        x2, y2 = locs2[i]
        A[2 * i, :] = np.array([-x1, -y1, -1, 0, 0, 0, x1 * x2, y1 * x2, x2])
        A[2 * i + 1, :] = np.array([0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2, y2])
    _, _, V = np.linalg.svd(A)
    H = V[-1, :].reshape((3, 3))
    H /= H[2, 2]
    return H

def applyH(H, locs):
    # Add a column of ones to locs
    ones = np.ones((locs.shape[0], 1))
    locs_hom = np.hstack((locs, ones))

    # Apply homography
    transformed = np.dot(H, locs_hom.T).T

    # Normalize homogeneous coordinates
    normalized = transformed[:, :2] / transformed[:, 2].reshape((-1, 1))

    return normalized

def compositeH(H, template, img):

    # Create a compositie image after warping the template image on top
    # of the image using homography

    img_shape = img.shape[1], img.shape[0]
    h, w = template.shape[:2]

    #Create mask of same size as template
    mask = np.ones((h, w))

    #Warp mask by appropriate homography
    warp_mask = cv2.warpPerspective(mask, H, img_shape)

    #Warp template by appropriate homography
    warp_template = cv2.warpPerspective(template, H, img_shape)

    #Use mask to combine the warped template and the image
    composite_img = img.copy()
    composite_img[warp_mask == 1] = warp_template[warp_mask == 1]


    return composite_img
