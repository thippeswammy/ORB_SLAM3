import cv2
import numpy as np


class FeatureTracker:
    """
    Finds indices of bad features (with status=0 or position outside the frame).

    Args:
        features: Detected (or tracked) features.
        frame: Image matrix.
        status: Vector of status flags returned by optical flow.

    Returns:
        Vector containing indices that should be filtered out.
    """

    def calcWrongFeatureIndices(self, features, frame, status):
        """Identify indices of features that are out of bounds.
        
        This function checks each feature point in the `features` list to determine if
        it lies within the valid bounds of the given `frame`. If a feature point is
        found to be out of bounds (either negative or exceeding the frame dimensions),
        its corresponding status is updated to indicate it as invalid. The function
        then returns the indices of all features that are marked as invalid.
        
        Args:
            features: A list of feature points, where each point is a tuple of (x, y) coordinates.
            frame: A numpy array representing the frame dimensions.
            status: A list or array indicating the validity of each feature point.
        """
        status_ = status.copy()
        for idx, pt in enumerate(features):
            if pt[0] < 0 or pt[1] < 0 or pt[0] > frame.shape[1] or pt[1] > frame.shape[0]:
                status_[idx] = 0
        wrongIndices = np.where(status_ == 0)[0]
        return wrongIndices

    """
    Tracks features using Lucas-Kanade optical flow and filters out bad features.

    Args:
        prevFrame: Previous image.
        currFrame: Current (next) image.
        prevPts: Features detected on previous frame.
        removeOutliers: Set to true if you want to remove bad features after tracking.

    Returns:
        Features from previous and current frame (tracked), both filtered.
    """

    def trackFeatures(self, prevFrame, currFrame, prevPts, removeOutliers=False):
        # Feature tracking on the 2nd frame
        currPts, status, _ = cv2.calcOpticalFlowPyrLK(prevFrame, currFrame, prevPts, None)

        if removeOutliers:
            # Filter out features that were not tracked (status=0) or are outside the image
            wrongIndices = self.calcWrongFeatureIndices(currPts, currFrame, status)
            prevPts = np.delete(prevPts, wrongIndices, axis=0)
            currPts = np.delete(currPts, wrongIndices, axis=0)

        return prevPts, currPts