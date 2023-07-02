## Lane Detection in Videos

This repository contains Python code for detecting and tracking lane lines in videos using computer vision techniques. The code processes each frame of a video, applies perspective transformation and color filtering to isolate the lane lines, and then uses a sliding window technique to detect and track the lines. The detected lines are then visualized on the original video frames.

## Requirements

Python 3.x
OpenCV (cv2)
NumPy
Matplotlib

## Customization

You can customize the code to suit your needs. Here are a few possible modifications:

Video file: If you want to use a different video file, you can replace the video2.mp4 file with your own video. Make sure to update the videoName variable in the code accordingly.

Perspective transformation: The code performs a perspective transformation to obtain a bird's-eye view of the road. If the current transformation doesn't work well for your video, you can modify the src1 and src2 arrays in the code to define new source points for the transformation.

Color filtering: The code filters the image by color to isolate the lane lines. You can adjust the color ranges in the filterByColor function to better detect the lane lines in your video.

Lane detection algorithm: The code currently uses a sliding window technique to detect and track the lane lines. If you want to use a different algorithm or improve the existing one, you can modify the slidingWindow function.
