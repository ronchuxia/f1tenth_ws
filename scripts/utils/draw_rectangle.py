"""
Draw a filled rectangle into a PGM map at pixel coordinates.

Pixel origin is the top-left of the image, +x right, +y down.
Default fill is 0 (occupied/black under map_server with negate=0).

Usage:
    python draw_rectangle.py <map.pgm> --x 120 --y 80 --w 30 --h 15
    python draw_rectangle.py <map.pgm> --x 120 --y 80 --w 30 --h 15 --yaw 30
    python draw_rectangle.py <map.pgm> --x 120 --y 80 --w 30 --h 15 --in-place
"""

import argparse
import os

import numpy as np
from PIL import Image, ImageDraw


def rectangle_corners(cx, cy, w, h, yaw_rad):
    hx, hy = w / 2.0, h / 2.0
    local = np.array([[ hx,  hy], [-hx,  hy], [-hx, -hy], [ hx, -hy]])
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    R = np.array([[c, -s], [s, c]])
    return [tuple(p) for p in (local @ R.T) + np.array([cx, cy])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('pgm', help='Path to PGM map')
    p.add_argument('--x', type=float, required=True, help='Rectangle center x (px)')
    p.add_argument('--y', type=float, required=True, help='Rectangle center y (px)')
    p.add_argument('--w', type=float, required=True, help='Rectangle width  (px)')
    p.add_argument('--h', type=float, required=True, help='Rectangle height (px)')
    p.add_argument('--yaw', type=float, default=0.0, help='Rotation in degrees (CW in image frame)')
    p.add_argument('--value', type=int, default=0, help='Pixel value to fill (0=occupied, 254=free)')
    p.add_argument('--out', default=None, help='Output PGM path. Default: <name>_rect.pgm')
    p.add_argument('--in-place', action='store_true', help='Overwrite the input PGM (ignores --out).')
    args = p.parse_args()

    in_path = os.path.abspath(args.pgm)
    img = Image.open(in_path)
    if img.mode != 'L':
        img = img.convert('L')

    corners = rectangle_corners(args.x, args.y, args.w, args.h, np.deg2rad(args.yaw))
    ImageDraw.Draw(img).polygon(corners, fill=int(args.value))

    if args.in_place:
        out_path = in_path
    elif args.out is not None:
        out_path = os.path.abspath(args.out)
    else:
        base, ext = os.path.splitext(in_path)
        out_path = base + '_rect' + ext

    img.save(out_path)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
