import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_image(path):
    """
    Reads an image from a given path and ensures it is loaded correctly.
    """
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image at path '{path}' could not be loaded.")
    return image


def compute_epipolar_lines(F_matrix, points):
    """
    Compute epipolar lines for given points using the fundamental matrix.

    :param F_matrix: Fundamental matrix.
    :param points: Points in homogeneous coordinates.
    :return: Epipolar lines in homogeneous coordinates.
    """
    return np.dot(F_matrix, points.T).T


def normalize_homogeneous_coords(coords):
    """
    Normalize homogeneous coordinates to convert them to Cartesian coordinates.
    """
    return coords[:, :2] / coords[:, 2:]


def find_intersection_points(lines1, lines2):
    """
    Find intersection points of two sets of epipolar lines.

    :param lines1: Epipolar lines from Image 1.
    :param lines2: Epipolar lines from Image 2.
    :return: Intersection points in Cartesian coordinates.
    """
    intersections = np.cross(lines1, lines2)
    return normalize_homogeneous_coords(intersections)


def plot_images_with_epipolar_lines(im1, im2, im3, points_im1, points_im2, lines_im3_1, lines_im3_2, intersections):
    """
    Plot images, selected points, and epipolar lines with intersections.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8))

    # Assign colors for corresponding points
    point_colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    # Plot Image 1 with points
    ax1.imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
    for i, color in enumerate(point_colors):
        ax1.scatter(points_im1[i, 0], points_im1[i, 1], c=color, marker='x', label=f'Point {i + 1}')
    ax1.set_title('Image 1')

    # Plot Image 2 with points
    ax2.imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
    for i, color in enumerate(point_colors):
        ax2.scatter(points_im2[i, 0], points_im2[i, 1], c=color, marker='x', label=f'Point {i + 1}')
    ax2.set_title('Image 2')

    # Plot Image 3 with epipolar lines and intersections
    ax3.imshow(cv2.cvtColor(im3, cv2.COLOR_BGR2RGB))
    for i, color in enumerate(point_colors):
        x = np.array([0, im3.shape[1]])  # Line endpoints in x
        y1 = - (lines_im3_1[i, 0] * x + lines_im3_1[i, 2]) / lines_im3_1[i, 1]
        y2 = - (lines_im3_2[i, 0] * x + lines_im3_2[i, 2]) / lines_im3_2[i, 1]

        ax3.plot(x, y1, color, linestyle='--', label=f"Point {i + 1} - Image 1")
        ax3.plot(x, y2, color, linestyle='-', label=f"Point {i + 1} - Image 2")

    # Plot intersection points
    ax3.scatter(intersections[:, 0], intersections[:, 1], c='black', marker='x', label='Intersection Points', s=100)
    ax3.set_title('Image 3')

    # Add legends and adjust layout
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    ax3.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# Main execution
if __name__ == "__main__":
    # Load images
    im1 = read_image('florence1.jpg')
    im2 = read_image('florence2.jpg')
    im3 = read_image('florence3.jpg')

    # Fundamental matrices
    Fmatrix13 = np.array([
        [6.04444985855117e-08, 2.56726410274219e-07, -0.000602529673152695],
        [2.45555247713476e-07, -8.38811736871429e-08, -0.000750892330636890],
        [-0.000444464396704832, 0.000390321707113558, 0.999999361609429]
    ])

    Fmatrix23 = np.array([
        [3.03994528999160e-08, 2.65672654114295e-07, -0.000870550254997210],
        [4.67606901933558e-08, -1.11709498607089e-07, -0.00169128012255720],
        [-1.38310618285550e-06, 0.00140690091935593, 0.999997201170569]
    ])

    # Selected points in Image 1 and Image 2
    selected_points_im1 = np.array([
        [676.0, 246.2],
        [626.6, 551.3],
        [983.5, 937.1],
        [808.65, 45.01],
        [651.60, 1576.52]
    ])

    selected_points_im2 = np.array([
        [530.4, 718.1],
        [474.05, 983.08],
        [758.7, 1173.0],
        [1191.36, 111.31],
        [463.507, 1712.526]
    ])

    # Convert to homogeneous coordinates
    homogeneous_points_im1 = np.hstack((selected_points_im1, np.ones((selected_points_im1.shape[0], 1))))
    homogeneous_points_im2 = np.hstack((selected_points_im2, np.ones((selected_points_im2.shape[0], 1))))

    # Compute epipolar lines in Image 3
    lines_im3_1 = compute_epipolar_lines(Fmatrix13, homogeneous_points_im1)
    lines_im3_2 = compute_epipolar_lines(Fmatrix23, homogeneous_points_im2)

    # Find intersection points of epipolar lines
    intersection_points = find_intersection_points(lines_im3_1, lines_im3_2)

    # Plot images and epipolar lines
    plot_images_with_epipolar_lines(im1, im2, im3, selected_points_im1, selected_points_im2, lines_im3_1, lines_im3_2,
                                    intersection_points)