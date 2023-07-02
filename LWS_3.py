import numpy as np
import cv2
from matplotlib import pyplot as plt

# Function to plot the area defined by points on the image


def plotArea(image, pts):
    for i in range(0, 4):
        cv2.circle(image, (pts[i, 0], pts[i, 1]),
                   radius=5, color=(255, 0, 0), thickness=-1)

    cv2.line(image, (pts[0, 0], pts[0, 1]),
             (pts[1, 0], pts[1, 1]), (0, 255, 0), thickness=3)
    cv2.line(image, (pts[1, 0], pts[1, 1]),
             (pts[2, 0], pts[2, 1]), (0, 255, 0), thickness=3)
    cv2.line(image, (pts[2, 0], pts[2, 1]),
             (pts[3, 0], pts[3, 1]), (0, 255, 0), thickness=3)
    cv2.line(image, (pts[3, 0], pts[3, 1]),
             (pts[0, 0], pts[0, 1]), (0, 255, 0), thickness=3)

# Function to put text information on the image


def putInfoImg(img, text, loc):
    cv2.putText(img,
                text,
                (loc[0], loc[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (0, 0, 255),
                2,
                cv2.LINE_4)

# Function to filter the image by color (white and yellow)


def filterByColor(image):
    imageHLS = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    # Define color ranges for white and yellow
    white_lower = np.array([0, 200, 0])
    white_upper = np.array([180, 255, 255])
    white_mask = cv2.inRange(imageHLS, white_lower, white_upper)

    yellow_lower = np.array([20, 0, 100])
    yellow_upper = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(imageHLS, yellow_lower, yellow_upper)

    # Combine white and yellow masks
    mask = cv2.bitwise_or(white_mask, yellow_mask)

    # Apply the mask to the image
    filteredImage = cv2.bitwise_and(image, image, mask=mask)

    return filteredImage, white_mask, yellow_mask, mask

# Function to find the two peaks in the bottom half of the binary image


def getTwoPeaks(binary_img):
    bottom = binary_img[height//2:, :]
    suma = np.sum(bottom, axis=0)

    x1 = np.argmax(suma)

    # Define the search range for the second peak
    x1_1 = max(0, x1-150)
    x1_2 = min(x1+150, len(suma)-1)

    # Exclude the first peak and find the second peak
    suma[x1_1:x1_2] = 0

    x2 = np.argmax(suma)

    if x1 < x2:
        xl, xr = x1, x2
    else:
        xl, xr = x2, x1

    return xl, xr

# Function to display the original image with the lane area outlined


def displayOriginal(original_img, left_x, right_x, M):
    lane = np.array([
        [left_x, 0],
        [left_x, 720],
        [right_x, 720],
        [right_x, 0]],
        dtype=np.float32)

    dst_lane = cv2.perspectiveTransform(np.array([lane]), M)
    plotArea(original_img, dst_lane[0].astype(np.int32))

# Function to display the warped image with the lane area outlined


def displayWarped(img, left_x, right_x):
    lane = np.array([
        [left_x, 0],
        [left_x, 720],
        [right_x, 720],
        [right_x, 0]],
        dtype=np.int32)

    plotArea(img, lane)


def slidingWindow(mask, xl, xr, warped_image):
    """
    Performs sliding window lane detection on the given binary mask and returns the left and right lane lines.

    Args:
        mask (numpy.ndarray): Binary mask of the lane lines.
        xl (int): Starting x-coordinate for the left lane line.
        xr (int): Starting x-coordinate for the right lane line.
        warped_image (numpy.ndarray): Warped image.

    Returns:
        tuple: Tuple containing the left and right lane lines.
    """

    def get_lines(mask, xl, warped_image):
        """
        Helper function to detect lane lines using sliding windows.

        Args:
            mask (numpy.ndarray): Binary mask of the lane lines.
            xl (int): Starting x-coordinate for the lane line.
            warped_image (numpy.ndarray): Warped image.

        Returns:
            numpy.ndarray: Detected lane line.
        """
        height = mask.shape[0]

        window_size_y = 30
        window_size_x = 90

        noWindows = height//(window_size_y*2)

        y_pos = height - window_size_y
        x_pos = xl

        x_last = x_pos
        y_last = y_pos

        leftPtsX = np.empty((0, 1), np.int32)
        leftPtsY = np.empty((0, 1), np.int32)

        for i in range(0, noWindows):
            cv2.rectangle(warped_image,
                          (x_pos-window_size_x, y_pos-window_size_y),
                          (x_pos+window_size_x, y_pos+window_size_y),
                          (255, 0, 0), 3)

            y, x = np.where(mask[y_pos-window_size_y:y_pos+window_size_y,
                                 x_pos-window_size_x:x_pos+window_size_x] == 255)

            if x.size > 0 and y.size > 0:
                x_reshaped = np.reshape(x+x_pos-window_size_x, (len(x), 1))
                leftPtsX = np.append(leftPtsX, x_reshaped, axis=0)
                y_reshaped = np.reshape(y+y_pos-window_size_y, (len(y), 1))
                leftPtsY = np.append(leftPtsY, y_reshaped, axis=0)

                x_pos = int(np.mean(x)) + x_pos-window_size_x
            else:
                x_pos = x_last

            y_pos = y_last - (window_size_y*2)
            x_last = x_pos
            y_last = y_pos

        if leftPtsX.shape[0] >= 3:
            line = np.polyfit(leftPtsY[:, 0], leftPtsX[:, 0], 2)
        else:
            line = np.zeros((3, 1))
        return line

    # Get the left and right lane lines
    left_line = get_lines(mask, xl, warped_image)
    right_line = get_lines(mask, xr, warped_image)
    return left_line, right_line


def plotPolyLine(warped_image, left_line, right_line):
    """
    Draws the left and right lane lines on the warped image.

    Args:
        warped_image (numpy.ndarray): Warped image.
        left_line (numpy.ndarray): Left lane line.
        right_line (numpy.ndarray): Right lane line.

    Returns:
        tuple: Tuple containing the left and right lane line points.
    """

    def draw_lines(line):
        """
        Helper function to draw a lane line on the image.

        Args:
            line (numpy.ndarray): Lane line coefficients.

        Returns:
            numpy.ndarray: Array of lane line points.
        """
        y_cords = np.linspace(
            0, warped_image.shape[0] - 1, warped_image.shape[0])
        x_cords = line[0] * (y_cords**2) + line[1] * y_cords + line[2]

        y_cords, x_cords = y_cords.astype(np.uint32), x_cords.astype(np.uint32)

        try:
            for x, y in zip(x_cords, y_cords):
                cv2.line(warped_image, (x - 4, y), (x + 4, y), (0, 255, 0), 2)
        except:
            pass
        points = np.array([x_cords, y_cords]).T
        return points

    left_points = draw_lines(left_line)
    right_points = draw_lines(right_line)

    return left_points, right_points


def draw_left_right_lines(frame, M_inv, left_points, right_points):
    """
    Draws the left and right lane lines on the original frame.

    Args:
        frame (numpy.ndarray): Original frame.
        M_inv (numpy.ndarray): Inverse perspective transformation matrix.
        left_points (numpy.ndarray): Left lane line points.
        right_points (numpy.ndarray): Right lane line points.

    Returns:
        numpy.ndarray: Frame with the lane lines drawn.
    """
    if np.any(left_points[0][0] < 1 or right_points[0][0] < 1 or left_points[0][0] == 0 or right_points[0][0] == 0):
        putInfoImg(frame, "Upozorenje!!! Prijelaz Linije.",
                   (int(frame.shape[0]//2), 150))
    else:
        left_pts, right_pts = np.float32(
            np.array([left_points])), np.float32(np.array([right_points]))
        left_pts_transformed, right_pts_transformed = cv2.perspectiveTransform(
            left_pts, M_inv), cv2.perspectiveTransform(right_pts, M_inv)

        points = np.concatenate(
            (left_pts_transformed, np.flip(right_pts_transformed, axis=1)), axis=1)

        mask = np.zeros_like(frame)

        points = [np.int32(points)]
        cv2.fillPoly(mask, points, (0, 255, 0))

        frame = cv2.bitwise_or(frame, mask)

    return frame


# main video
videoName = 'video1.mp4'
cap = cv2.VideoCapture(videoName)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define source and destination points for perspective transformation
src1 = np.array([
    [466, 510],
    [181, 670],
    [1334, 670],
    [890, 510]
], dtype=np.float32)

src2 = np.array([
    [550, 448],
    [219, 600],
    [1082, 605],
    [801, 442]
], dtype=np.float32)

dst = np.array([
    [0, 0],
    [0, height],
    [width, height],
    [width, 0]
], dtype=np.float32)

M = cv2.getPerspectiveTransform(src2, dst)
M_inv = cv2.getPerspectiveTransform(dst, src2)

k = 1
time = 1


# Loop through each frame in the video
while (True):
    e1 = cv2.getTickCount()

    ret, frame = cap.read()

    if not ret:
        break

    # Warp the frame using perspective transformation
    warped_image = cv2.warpPerspective(frame, M, (width, height))

    # Filter the warped image by color
    filteredImage, white_mask, yellow_mask, mask = filterByColor(warped_image)

    # Find the two peaks in the binary mask
    xl, xr = getTwoPeaks(mask)

    # Apply sliding window technique to find the left and right lines
    left_line, right_line = slidingWindow(mask, xl, xr, warped_image)

    # Plot the left and right lines on the warped image
    left_points, right_points = plotPolyLine(
        warped_image, left_line, right_line)

    # Draw the left and right lines on the original frame
    lines_on_frame = draw_left_right_lines(
        frame, M_inv, left_points, right_points)

    cv2.imshow("frame", lines_on_frame)

    # Pause the video and quit if 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        while True:
            key2 = cv2.waitKey(1) or 0xff
            if key2 == ord('p'):
                break
            if key2 == ord('q'):
                break

    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency()


cap.release()
cv2.destroyAllWindows()
