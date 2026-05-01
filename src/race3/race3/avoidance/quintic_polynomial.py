"""
Quintic polynomial with endpoint boundary conditions on value, first, and
second derivatives at both ends.

Ported from
reference/stack/PythonRobotics/PathPlanning/QuinticPolynomialsPlanner/
quintic_polynomials_planner.py (class QuinticPolynomial, lines 25-66).

The parameter is called `time` to match the reference, but it is just the
horizon in whatever independent variable the caller chose. For our d(s)
Frenet planner we pass arc length.
"""

import numpy as np


class QuinticPolynomial:
    def __init__(self, xs, vxs, axs, xe, vxe, axe, time):
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        T = time
        A = np.array([[T ** 3,     T ** 4,      T ** 5],
                      [3 * T ** 2, 4 * T ** 3,  5 * T ** 4],
                      [6 * T,      12 * T ** 2, 20 * T ** 3]])
        b = np.array([xe - self.a0 - self.a1 * T - self.a2 * T ** 2,
                      vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)
        self.a3, self.a4, self.a5 = x[0], x[1], x[2]

    def calc_point(self, t):
        return (self.a0 + self.a1 * t + self.a2 * t ** 2
                + self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5)

    def calc_first_derivative(self, t):
        return (self.a1 + 2 * self.a2 * t + 3 * self.a3 * t ** 2
                + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4)

    def calc_second_derivative(self, t):
        return (2 * self.a2 + 6 * self.a3 * t
                + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3)

    def calc_third_derivative(self, t):
        return 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2
