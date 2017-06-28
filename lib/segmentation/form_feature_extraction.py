from __future__ import division

import re

from cached_property import cached_property
import numpy as np
from numpy import pi as PI
import numpy_groupies as npg
import scipy.ndimage as ndi


class FormFeatureExtraction(object):
    def __init__(self, segmentation):
        self.segmentation = segmentation

    def get_features(self, features=None):
        methods = FormFeatureExtraction.methods

        if features is None:
            features = range(len(methods))

        methods = [methods[i] for i in features]

        return np.stack([getattr(self, m) for m in methods]).T

    @cached_property
    def _flat(self):
        return self.segmentation.flatten()

    @cached_property
    def _ys(self):
        height, width = self.segmentation.shape
        return np.arange(height).repeat(width)

    @cached_property
    def _xs(self):
        height, width = self.segmentation.shape
        return np.tile(np.arange(width), height)

    @cached_property
    def _ys2(self):
        return self._ys * self._ys

    @cached_property
    def _xs2(self):
        return self._xs * self._xs

    @cached_property
    def _group_idx(self):
        return np.arange(np.unique(self._flat).size)

    @cached_property
    def _extrema_y(self):
        # npg aggregate min/max is very slow, use scipy.ndimage instead.
        mi, ma, _, _ = ndi.extrema(self._ys, self._flat, self._group_idx)
        return mi, ma

    @cached_property
    def _extrema_x(self):
        # npg aggregate min/max is very slow, use scipy.ndimage instead.
        mi, ma, _, _ = ndi.extrema(self._xs, self._flat, self._group_idx)
        return mi, ma

    @cached_property
    def _min_y(self):
        return self._extrema_y[0]

    @cached_property
    def _max_y(self):
        return self._extrema_y[1] + 1

    @cached_property
    def _min_x(self):
        return self._extrema_x[0]

    @cached_property
    def _max_x(self):
        return self._extrema_x[1] + 1

    @cached_property
    def _M_00(self):
        ones = np.ones_like(self._flat)
        return npg.aggregate(self._flat, ones, func='sum')

    @cached_property
    def _M_01(self):
        return npg.aggregate(self._flat, self._ys, func='sum')

    @cached_property
    def _M_10(self):
        return npg.aggregate(self._flat, self._xs, func='sum')

    @cached_property
    def _M_11(self):
        return npg.aggregate(self._flat, self._ys * self._xs, func='sum')

    @cached_property
    def _M_02(self):
        return npg.aggregate(self._flat, self._ys2, func='sum')

    @cached_property
    def _M_20(self):
        return npg.aggregate(self._flat, self._xs2, func='sum')

    @cached_property
    def _M_12(self):
        return npg.aggregate(self._flat, self._ys2 * self._xs, func='sum')

    @cached_property
    def _M_21(self):
        return npg.aggregate(self._flat, self._ys * self._xs2, func='sum')

    @cached_property
    def _M_03(self):
        return npg.aggregate(self._flat, self._ys2 * self._ys, func='sum')

    @cached_property
    def _M_30(self):
        return npg.aggregate(self._flat, self._xs2 * self._xs, func='sum')

    @cached_property
    def _centroid_y(self):
        return self._M_01 / self._M_00

    @cached_property
    def _centroid_x(self):
        return self._M_10 / self._M_00

    @cached_property
    def _centroid_y2(self):
        return self._centroid_y * self._centroid_y

    @cached_property
    def _centroid_x2(self):
        return self._centroid_x * self._centroid_x

    @cached_property
    def _inertia_tensor_eigvals_sum_1(self):
        return (self.inertia_tensor_20 + self.inertia_tensor_02) / 2

    @cached_property
    def _inertia_tensor_eigvals_sum_2(self):
        sum_1 = 4 * self.inertia_tensor_11 * self.inertia_tensor_11
        sum_2 = (self.inertia_tensor_20 - self.inertia_tensor_02)
        sum_2 = sum_2 * sum_2
        return np.sqrt(sum_1 + sum_2) / 2

    def _nu_denominator(self, i, j):
        return np.power(self._M_00, 1 + (i + j) / 2)

    @cached_property
    def mu_11(self):
        return self._M_11 - self._centroid_x * self._M_01

    @cached_property
    def mu_02(self):
        return self._M_02 - self._centroid_y * self._M_01

    @cached_property
    def mu_20(self):
        return self._M_20 - self._centroid_x * self._M_10

    @cached_property
    def mu_12(self):
        sum_1 = self._M_12 - 2 * self._centroid_y * self._M_11
        sum_2 = self._centroid_x * self._M_02
        sum_3 = 2 * self._centroid_y2 * self._M_10
        return sum_1 - sum_2 + sum_3

    @cached_property
    def mu_21(self):
        sum_1 = self._M_21 - 2 * self._centroid_x * self._M_11
        sum_2 = self._centroid_y * self._M_20
        sum_3 = 2 * self._centroid_x2 * self._M_01
        return sum_1 - sum_2 + sum_3

    @cached_property
    def mu_03(self):
        sum_1 = 3 * self._centroid_y * self._M_02
        sum_2 = 2 * self._centroid_y2 * self._M_01
        return self._M_03 - sum_1 + sum_2

    @cached_property
    def mu_30(self):
        sum_1 = 3 * self._centroid_x * self._M_20
        sum_2 = 2 * self._centroid_x2 * self._M_10
        return self._M_30 - sum_1 + sum_2

    @cached_property
    def inertia_tensor_02(self):
        return self.mu_02 / self._M_00

    @cached_property
    def inertia_tensor_20(self):
        return self.mu_20 / self._M_00

    @cached_property
    def inertia_tensor_11(self):
        return self.mu_11 / self._M_00

    @cached_property
    def inertia_tensor_eigvals_1(self):
        a = self._inertia_tensor_eigvals_sum_1
        b = self._inertia_tensor_eigvals_sum_2
        return a + b

    @cached_property
    def inertia_tensor_eigvals_2(self):
        a = self._inertia_tensor_eigvals_sum_1
        b = self._inertia_tensor_eigvals_sum_2
        return a - b

    @cached_property
    def nu_11(self):
        return self.mu_11 / self._nu_denominator(1, 1)

    @cached_property
    def nu_02(self):
        return self.mu_02 / self._nu_denominator(0, 2)

    @cached_property
    def nu_20(self):
        return self.mu_20 / self._nu_denominator(2, 0)

    @cached_property
    def nu_12(self):
        return self.mu_12 / self._nu_denominator(1, 2)

    @cached_property
    def nu_21(self):
        return self.mu_21 / self._nu_denominator(2, 1)

    @cached_property
    def nu_03(self):
        return self.mu_03 / self._nu_denominator(0, 3)

    @cached_property
    def nu_30(self):
        return self.mu_30 / self._nu_denominator(3, 0)

    @cached_property
    def hu_1(self):
        return self.nu_20 + self.nu_02

    @cached_property
    def hu_2(self):
        sum_1 = self.nu_20 - self.nu_02
        sum_1 = sum_1 * sum_1
        sum_2 = 2 * self.nu_11
        sum_2 = sum_2 * sum_2
        return sum_1 + sum_2

    @cached_property
    def hu_3(self):
        sum_1 = self.nu_30 - 3 * self.nu_12
        sum_1 = sum_1 * sum_1
        sum_2 = 3 * self.nu_21 - self.nu_03
        sum_2 = sum_2 * sum_2
        return sum_1 + sum_2

    @cached_property
    def hu_4(self):
        sum_1 = self.nu_30 + self.nu_12
        sum_1 = sum_1 * sum_1
        sum_2 = self.nu_21 + self.nu_03
        sum_2 = sum_2 * sum_2
        return sum_1 + sum_2

    @cached_property
    def hu_5(self):
        prod_1 = (self.nu_30 - 3 * self.nu_12) * (self.nu_30 + self.nu_12)
        prod_21 = self.nu_30 + self.nu_12
        prod_21 = prod_21 * prod_21
        prod_22 = self.nu_21 + self.nu_03
        prod_22 = 3 * prod_22 * prod_22
        prod_2 = prod_21 - prod_22
        sum_1 = prod_1 * prod_2
        prod_1 = (3 * self.nu_21 - self.nu_03) * (self.nu_21 + self.nu_03)
        prod_21 = self.nu_30 + self.nu_12
        prod_21 = 3 * prod_21 * prod_21
        prod_22 = self.nu_21 + self.nu_03
        prod_22 = prod_22 * prod_22
        prod_2 = prod_21 - prod_22
        sum_2 = prod_1 * prod_2
        return sum_1 + sum_2

    @cached_property
    def hu_6(self):
        prod_1 = self.nu_20 - self.nu_02
        prod_21 = self.nu_30 + self.nu_12
        prod_21 = prod_21 * prod_21
        prod_22 = self.nu_21 + self.nu_03
        prod_22 = prod_22 * prod_22
        sum_1 = prod_1 * (prod_21 - prod_22)
        sum_2 = 4 * self.nu_11 * (self.nu_30 + self.nu_12)
        sum_2 *= (self.nu_21 + self.nu_03)
        return sum_1 + sum_2

    @cached_property
    def hu_7(self):
        prod_1 = (3 * self.nu_21 - self.nu_03) * (self.nu_30 + self.nu_12)
        prod_21 = self.nu_30 + self.nu_12
        prod_21 = prod_21 * prod_21
        prod_22 = self.nu_21 + self.nu_03
        prod_22 = 3 * prod_22 * prod_22
        sum_1 = prod_1 * (prod_21 - prod_22)
        prod_1 = (self.nu_30 - 3 * self.nu_12) * (self.nu_21 + self.nu_03)
        prod_21 = self.nu_30 + self.nu_12
        prod_21 = 3 * prod_21 * prod_21
        prod_22 = self.nu_21 + self.nu_03
        prod_22 = prod_22 * prod_22
        sum_2 = prod_1 * (prod_21 - prod_22)
        return sum_1 - sum_2

    @cached_property
    def area(self):
        return self._M_00

    @cached_property
    def bbox_height(self):
        return self._max_y - self._min_y

    @cached_property
    def bbox_width(self):
        return self._max_x - self._min_x

    @cached_property
    def bbox_area(self):
        return self.bbox_height * self.bbox_width

    @cached_property
    def centroid_y(self):
        return self._centroid_y - self._min_y

    @cached_property
    def centroid_x(self):
        return self._centroid_x - self._min_x

    @cached_property
    def eccentricity(self):
        eigval_1 = self.inertia_tensor_eigvals_1
        eigval_2 = self.inertia_tensor_eigvals_2
        with np.errstate(invalid='ignore'):
            result = np.sqrt(1 - eigval_2 / eigval_1)
        return np.nan_to_num(result)

    @cached_property
    def equivalent_diameter(self):
        return np.sqrt(4 * self._M_00 / PI)

    @cached_property
    def extent(self):
        return self._M_00 / self.bbox_area

    @cached_property
    def major_axis_length(self):
        return 4 * np.sqrt(self.inertia_tensor_eigvals_1)

    @cached_property
    def minor_axis_length(self):
        return 4 * np.sqrt(np.abs(self.inertia_tensor_eigvals_2))

    @cached_property
    def orientation(self):
        a = self.inertia_tensor_20
        b = self.inertia_tensor_11
        c = self.inertia_tensor_02

        right = a - c
        result = -0.5 * np.arctan2(2 * b, right)
        b_new = np.where(b > 0, -np.pi / 4, np.pi / 4)
        return np.where(right == 0, b_new, result)


methods = dir(FormFeatureExtraction)
# Filter private properties and get_features method.
p = re.compile('^_|(get_features)')
methods = [m for m in methods if not p.match(m)]
methods = sorted(methods)
FormFeatureExtraction.methods = methods
