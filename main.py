import numpy as np
from scipy.optimize import least_squares


def triangulate_position(point1, point2, P1, P2):
    """
    Triangulate the 3D position of a point from its 2D projections in two camera images.

    Args:
    - point1: 2D coordinates in the first camera image.
    - point2: 2D coordinates in the second camera image.
    - P1: Projection matrix for the first camera.
    - P2: Projection matrix for the second camera.

    Returns:
    - Estimated 3D position of the point.
    """

    def triangulate_func(X, P1, P2, point1, point2):
        # Convert X to homogeneous coordinates for projection
        X_homog = np.append(X, 1)

        # Project X onto both image planes
        proj1 = P1 @ X_homog
        proj2 = P2 @ X_homog

        # Check for near-zero values and adjust to avoid division by zero
        epsilon = 1e-6
        if abs(proj1[2]) < epsilon or abs(proj2[2]) < epsilon:
            # Return a large residual instead of np.inf
            return np.full(
                4, 1e6
            )  # Adjust the size based on the expected residual size

        # Convert to Cartesian coordinates
        proj1_cartesian = proj1[:2] / proj1[2]
        proj2_cartesian = proj2[:2] / proj2[2]

        # Compute residuals
        residual1 = proj1_cartesian - point1
        residual2 = proj2_cartesian - point2

        return np.concatenate([residual1, residual2])

    # Initial guess for the 3D point position
    X0 = [0, 0, 0]

    # Perform least squares optimization to minimize the reprojection error
    result = least_squares(triangulate_func, X0, args=(P1, P2, point1, point2))

    if not result.success:
        raise ValueError("Triangulation optimization failed to converge.")

    return result.x


point1 = [
    100,
    150,
]  # BOILER- Replace these values with the actual 2D coordinates from camera 1
point2 = [
    200,
    250,
]  # BOILER- Replace these values with the actual 2D coordinates from camera 2
P1 = np.array(
    [
        [1000, 0, 320, 0],  # fx, 0, cx, tx
        [0, 1000, 240, 0],  # 0, fy, cy, ty
        [0, 0, 1, 0],  # 0, 0, 1, tz
    ]
)

P2 = np.array(
    [
        [1000, 0, 320, 0],  # fx, 0, cx, tx
        [0, 1000, 240, 0],  # 0, fy, cy, ty
        [0, 0, 1, 0],  # 0, 0, 1, tz
    ]
)

try:
    position_3d = triangulate_position(point1, point2, P1, P2)
    print(f"Triangulated 3D position: {position_3d}")
except ValueError as e:
    print(e)
