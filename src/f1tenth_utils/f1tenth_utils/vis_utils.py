import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from nav_msgs.msg import OccupancyGrid


def visualize_point(point, stamp, frame_id='/map', ns='point', id=0, color=(1.0, 0.0, 0.0, 1.0)):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = ns
    marker.id = id
    marker.type = Marker.SPHERE

    marker.pose.position.x = point[0]
    marker.pose.position.y = point[1]
    marker.pose.position.z = 0.0

    marker.scale.x = 0.2
    marker.scale.y = 0.2
    marker.scale.z = 0.2

    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]
    return marker


def visualize_points(points, stamp, frame_id='/map', ns='points', id=0, color=(1.0, 0.0, 0.0, 1.0), color_end=None):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = ns
    marker.id = id
    marker.type = Marker.POINTS
    marker.action = Marker.ADD

    for point in points:
        marker.points.append(Point(x=point[0], y=point[1], z=0.0))

    marker.scale.x = 0.1
    marker.scale.y = 0.1

    if color_end is None:
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
    else:
        n = max(len(points) - 1, 1)
        for i in range(len(points)):
            t = i / n
            marker.colors.append(ColorRGBA(
                r=color[0] + t * (color_end[0] - color[0]),
                g=color[1] + t * (color_end[1] - color[1]),
                b=color[2] + t * (color_end[2] - color[2]),
                a=color[3] + t * (color_end[3] - color[3]),
            ))

    return marker


def visualize_occupancy_grid(occupancy_grid, stamp, resolution, width, height, origin_x_idx, origin_y_idx, frame_id='/ego_racecar/base_link'):
    msg = OccupancyGrid()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.info.resolution = resolution
    msg.info.width = width
    msg.info.height = height
    msg.info.origin.position.x = -origin_x_idx * resolution
    msg.info.origin.position.y = -origin_y_idx * resolution
    msg.info.origin.position.z = 0.0

    msg.data = (occupancy_grid.T.flatten() * 100).astype(int).tolist()  # flatten and convert to percentage
    return msg
        

def visualize_occupancy_grid_as_marker_array(occupancy_grid, stamp, resolution, origin_x_idx, origin_y_idx, frame_id='/ego_racecar/base_link', ns='occupancy_grid', id=0, color=(1.0, 0.0, 0.0, 0.8)):
    """
    Visualize an occupancy grid as a MarkerArray using a CUBE_LIST marker.
    Each occupied cell is rendered as a cube at its world-frame position.

    Args:
        occupancy_grid (np.ndarray): 2D array indexed [x_idx, y_idx], non-zero = occupied
        stamp: ROS timestamp
        resolution (float): meters per cell
        origin_x_idx (int): grid x-index corresponding to the frame origin
        origin_y_idx (int): grid y-index corresponding to the frame origin
        frame_id (str): coordinate frame
        ns (str): marker namespace
        id (int): marker id
        color (tuple): (r, g, b, a)
    Returns:
        MarkerArray
    """
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = ns
    marker.id = id
    marker.type = Marker.CUBE_LIST
    marker.action = Marker.ADD

    marker.scale.x = resolution
    marker.scale.y = resolution
    marker.scale.z = 0.05  # flat slab

    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]

    occupied = np.argwhere(occupancy_grid > 0)
    for x_idx, y_idx in occupied:
        p = Point()
        p.x = (x_idx - origin_x_idx) * resolution
        p.y = (y_idx - origin_y_idx) * resolution
        p.z = 0.0
        marker.points.append(p)

    marker_array = MarkerArray()
    marker_array.markers.append(marker)
    return marker_array


def visualize_path(path, stamp, frame_id='/ego_racecar/base_link', ns='path', id=0, color=(1.0, 0.0, 0.0, 1.0)):
    marker = Marker()
    marker.header.stamp = stamp
    marker.header.frame_id = frame_id
    marker.ns = ns
    marker.id = id
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD

    for point in path:
        marker.points.append(Point(x=point[0], y=point[1], z=0.0))

    marker.scale.x = 0.05

    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]

    return marker


def predict_trajectory(angle, steps=20, arc_length=2.0, wheelbase=0.3302):
    trajectory = []
    if abs(angle) < 1e-3:
        # straight line along +x
        for i in range(steps):
            trajectory.append([arc_length * i / (steps - 1), 0.0])
    else:
        R = wheelbase / np.tan(angle)
        total_angle = arc_length / R
        for i in range(steps):
            a = -np.pi / 2 + total_angle * i / (steps - 1)
            trajectory.append([R * np.cos(a), R + R * np.sin(a)])
    return np.array(trajectory)


def visualize_trajectory(angle, stamp, frame_id='/ego_racecar/base_link', ns='trajectory', id=0, color=(1.0, 1.0, 0.0, 1.0), steps=20, arc_length=2.0, wheelbase=0.3302):
    trajectory = predict_trajectory(angle, steps, arc_length, wheelbase=wheelbase)
    marker = visualize_path(trajectory, stamp=stamp, frame_id=frame_id, ns=ns, id=id, color=color)
    return marker


def visualize_tree(pos, parent, stamp, frame_id='/ego_racecar/base_link', ns='tree', id=0, color=(1.0, 0.0, 0.0, 1.0)):
    marker = Marker()
    marker.header.stamp = stamp
    marker.header.frame_id = frame_id
    marker.ns = ns
    marker.id = id
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD

    for i in range(1, len(pos)):
        p_idx = parent[i]
        p_parent = Point(x=pos[p_idx][0], y=pos[p_idx][1], z=0.0)
        p_child = Point(x=pos[i][0], y=pos[i][1], z=0.0)
        marker.points.append(p_parent)
        marker.points.append(p_child)

    marker.scale.x = 0.02

    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]

    return marker