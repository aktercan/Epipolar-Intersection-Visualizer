Epipolar Intersection Visualizer

This project provides a Python-based tool to compute and visualize epipolar geometry across three images. The tool uses fundamental matrices and user-selected points to calculate epipolar lines, find their intersections, and display these results on corresponding images.

Features

    •    Epipolar Line Computation: Calculates epipolar lines in the third image using fundamental matrices from the first and second images.
    •    Intersection Point Detection: Finds intersection points of epipolar lines from two sets of points.
    •    Visual Analysis: Displays images with selected points, epipolar lines, and their intersections for a comprehensive visual representation.

Why This Project?

Epipolar geometry is crucial in stereo vision and multi-view image analysis. This project demonstrates the principles of epipolar line computation and intersection detection, providing a practical and visual tool for educational purposes and advanced computer vision research.

How It Works

    1.    Image Input:
    •    Load three images representing stereo views.
    
    2.    Fundamental Matrices:
    •    Use two fundamental matrices to compute epipolar lines for points in the first and second images.
   
    3.    Epipolar Line Intersection:
    •    Find the intersection points of epipolar lines in the third image.
   
    4.    Visualization:
    •    Display:
    •    Selected points on the first and second images.
    •    Epipolar lines from both sets on the third image.
    •    Intersection points.

Requirements

To run the project, ensure you have the following installed:
    •    Python 3.8+
    •    Libraries:
    •    opencv-python
    •    numpy
    •    matplotlib
    
Example Output

    •    Image 1: Displays user-selected points.
    •    Image 2: Displays user-selected points.
    •    Image 3: Shows epipolar lines from Image 1 and Image 2, and their intersections.


