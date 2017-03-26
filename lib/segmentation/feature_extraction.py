import numpy as np
from skimage.measure import regionprops
from skimage import color
from sklearn.preprocessing import MinMaxScaler


def feature_extraction(segmentation, image):
    # We need to increment the segmentation, because labels with value 0 are
    # ignored when calling `regionprops`.
    segmentation += 1

    intensity_image = color.rgb2gray(image)
    props = regionprops(segmentation, intensity_image)

    NUM_FEATURES = 85
    features = np.zeros((len(props), NUM_FEATURES), dtype=np.float32)

    for i, prop in enumerate(props):
        area = prop['area']
        min_row, min_col, max_row, max_col = prop['bbox']
        bbox_height = max_row - min_row
        bbox_width = max_col - min_col
        bbox_area = bbox_height * bbox_width
        convex_area = prop['convex_area']
        eccentricity = prop['eccentricity']
        equivalent_diameter = prop['equivalent_diameter']
        extent = prop['extent']
        filled_area = prop['filled_area']
        major_axis_length = prop['major_axis_length']
        minor_axis_length = prop['minor_axis_length']
        diff_intensity = prop['max_intensity'] - prop['min_intensity']
        mean_intensity = prop['mean_intensity']
        orientation = prop['orientation']
        perimeter = prop['perimeter']
        solidity = prop['solidity']
        inertia_tensor = prop['inertia_tensor'].flatten()  # 4 values
        inertia_tensor_eigvals = prop['inertia_tensor_eigvals']  # 2 values
        local_centroid = prop['local_centroid']  # 2 values
        moments = prop['moments'].flatten()  # 16 values
        moments_central = prop['moments_central'].flatten()  # 16 values
        moments_hu = prop['moments_hu']  # 8 values
        weighted_moments = prop['weighted_moments'].flatten()  # 16 values

        sliced_image = image[min_row:max_row, min_col:max_col]
        sliced_image = sliced_image[prop['image']]
        mean_color_r = sliced_image[..., 0].mean()
        mean_color_g = sliced_image[..., 1].mean()
        mean_color_b = sliced_image[..., 2].mean()
        diff_color_r = sliced_image[..., 0].max() - sliced_image[..., 0].min()
        diff_color_g = sliced_image[..., 1].max() - sliced_image[..., 1].min()
        diff_color_b = sliced_image[..., 2].max() - sliced_image[..., 2].min()

        feature = [
            area, bbox_height, bbox_width, bbox_area, convex_area,
            eccentricity, equivalent_diameter, extent, filled_area,
            major_axis_length, minor_axis_length, diff_intensity,
            mean_intensity, orientation, perimeter, solidity
        ]  # bis index 15
        feature.extend(inertia_tensor)  # bis index 19
        feature.extend(inertia_tensor_eigvals)  # bis index 21
        feature.extend(local_centroid)  # bis index 23
        feature.extend(moments)  # bis index 39
        feature.extend(moments_central)  # bis index 55
        feature.extend(moments_hu)  # bis index 63
        feature.extend(weighted_moments)  # bis index 79
        feature.extend([mean_color_r, mean_color_g, mean_color_b])
        feature.extend([diff_color_r, diff_color_g, diff_color_b])

        assert len(feature) == NUM_FEATURES
        features[i] = np.array(feature)

    # Scale linear to [0, 1].
    return MinMaxScaler().fit_transform(features)
