import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import argparse


def main(file_name, subsample_rate=100, draw=False, smooth=False, num_points=100):
    out_dir = os.path.dirname(file_name)
    points = np.loadtxt(file_name, delimiter=',')
    points = points[::subsample_rate]   # shape (n, 2)

    # draw raw points
    if draw:
        fig = plt.figure()
        plt.scatter(points[:, 0], points[:, 1], s=2)
        plt.title('Raw Points')
        plt.axis('equal')
        fig.savefig(os.path.join(out_dir, 'raw.png'), dpi=300)

    if smooth:
        x = points[:, 0]
        y = points[:, 1]
        tck, u = splprep([x, y], s=0)

        u = np.linspace(0, 1, num_points)
        x_smooth, y_smooth = splev(u, tck)

        if draw:
            fig = plt.figure()
            plt.scatter(x_smooth, y_smooth, s=2)
            plt.title('Smoothed Points')
            plt.axis('equal')
            fig.savefig(os.path.join(out_dir, 'smooth.png'), dpi=300)

        points = np.stack([x_smooth, y_smooth], axis=1)

    np.savetxt(os.path.join(out_dir, 'waypoints_postprocessed.csv'), points, delimiter=',')

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', default='waypoints.csv', nargs='?')
    parser.add_argument('--subsample-rate', type=int, default=100)
    parser.add_argument('--draw', action='store_true', default=True)
    parser.add_argument('--no-draw', dest='draw', action='store_false')
    parser.add_argument('--smooth', action='store_true', default=True)
    parser.add_argument('--no-smooth', dest='smooth', action='store_false')
    parser.add_argument('--num-points', type=int, default=100)
    args = parser.parse_args()
    main(args.file_name, subsample_rate=args.subsample_rate, draw=args.draw, smooth=args.smooth, num_points=args.num_points)