import numpy as np
import math
from scipy.interpolate import CubicSpline
import itertools
from typing import List, Callable, Tuple, Union

# 1. Noise
def noise_transform_vectorized(X: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    """
    Adding random Gaussian noise with mean 0
    """
    noise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X + noise

# 2. Scaling
def scaling_transform_vectorized(X: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """
    Scaling by a random factor
    """
    scaling_factor = np.random.normal(loc=1.0, scale=sigma, size=(X.shape[0], 1, X.shape[2]))
    return X * scaling_factor

# 3. Rotation
def rotation_transform_vectorized(X: np.ndarray) -> np.ndarray:
    """
    Applies a different random 3D rotation to each sensor triplet (x, y, z).
    For 9-channel input [x1, y1, z1, x2, y2, z2, x3, y3, z3], each (x, y, z) group gets its own rotation.
    
    Args:
        X: Input data [batch_size, window_size, channels]
           Channel dimension should be multiple of 3 for sensor axes (x, y, z)
           
    Returns:
        X_rot: Rotated data with different rotation per sensor group
        
    Raises:
        ValueError: If channel dimension is not multiple of 3
    """
    batch_size, window_size, channels = X.shape
    
    if channels % 3 != 0:
        raise ValueError(f"Channel dimension ({channels}) must be multiple of 3 for proper sensor rotation")
    
    n_sensors = channels // 3
    X_rot = np.zeros_like(X)
    
    # For each sample in batch
    for b in range(batch_size):
        # For each sensor group
        for i in range(n_sensors):
            # Generate random rotation axis and angle for this sensor
            axis = np.random.uniform(low=-1, high=1, size=3)
            axis = axis / np.linalg.norm(axis)
            angle = np.random.uniform(low=-np.pi, high=np.pi)
            
            # Get rotation matrix
            R = axangle2mat(axis, angle, is_normalized=True)
            
            # Apply rotation to this sensor's channels
            sensor_channels = slice(i*3, (i+1)*3)
            vec = X[b, :, sensor_channels]  # [window_size, 3]
            X_rot[b, :, sensor_channels] = vec @ R  # Apply rotation
    
    return X_rot

def axangle2mat(axis, angle, is_normalized = False) -> np.ndarray:
    """
    Rotation matrix for rotation angle `angle` around `axis`
    Notes
    -----
    Reference: the Transforms3d package - transforms3d.axangles.axangle2mat
    From: http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    """
    x, y, z = axis
    if not is_normalized:
        n = math.sqrt(x*x + y*y + z*z)
        x = x/n
        y = y/n
        z = z/n
    c = math.cos(angle); s = math.sin(angle); C = 1-c
    xs = x*s;   ys = y*s;   zs = z*s
    xC = x*C;   yC = y*C;   zC = z*C
    xyC = x*yC; yzC = y*zC; zxC = z*xC
    return np.array([
            [ x*xC+c,   xyC-zs,   zxC+ys ],
            [ xyC+zs,   y*yC+c,   yzC-xs ],
            [ zxC-ys,   yzC+xs,   z*zC+c ]])

# 4. Negation
def negate_transform_vectorized(X: np.ndarray) -> np.ndarray:
    """
    Inverting the signals
    """
    return X * -1

# 5. Time Flip
def time_flip_transform_vectorized(X: np.ndarray) -> np.ndarray:
    """
    Reversing the direction of time
    """
    return X[:, ::-1, :]


# 6. Channel Shuffle
def channel_shuffle_transform_vectorized(X: np.ndarray) -> np.ndarray:
    """
    Shuffling the axes (x, y, z) within each sensor group while preserving sensor structure.
    For 9-channel input [x1, y1, z1, x2, y2, z2, x3, y3, z3], 
    each sensor's (x, y, z) axes are permuted independently.
    
    Args:
        X: Input data [batch_size, window_size, channels]
           Channel dimension should be multiple of 3 for sensor axes (x, y, z)
           
    Returns:
        X_shuffled: Channel shuffled data with sensor structure preserved
        
    Raises:
        ValueError: If channel dimension is not multiple of 3
    """
    batch_size, window_size, channels = X.shape
    
    if channels % 3 != 0:
        raise ValueError(f"Channel dimension ({channels}) must be multiple of 3 for proper sensor-aware channel shuffle")
    
    n_sensors = channels // 3
    X_shuffled = np.zeros_like(X)
    
    # For each sample in batch
    for b in range(batch_size):
        # For each sensor group
        for i in range(n_sensors):
            # Generate random permutation for this sensor's axes (x, y, z)
            sensor_permutation = np.random.permutation(3)  # [0,1,2] -> [1,2,0] etc.
            
            # Apply permutation to this sensor's channels
            sensor_channels = slice(i*3, (i+1)*3)
            original_sensor_data = X[b, :, sensor_channels]  # [window_size, 3]
            X_shuffled[b, :, sensor_channels] = original_sensor_data[:, sensor_permutation]
    
    return X_shuffled

# 7. Time Segment Permutation
def time_segment_permutation_transform_improved(X: np.ndarray, num_segments: int = 4) -> np.ndarray:
    """
    Randomly scrambling sections of the signal
    """
    segment_points_permuted = np.random.choice(X.shape[1], size=(X.shape[0], num_segments))
    segment_points = np.sort(segment_points_permuted, axis=1)

    X_transformed = np.empty(shape=X.shape)
    for i, (sample, segments) in enumerate(zip(X, segment_points)):
        # print(sample.shape)
        splitted = np.array(np.split(sample, np.append(segments, X.shape[1])))
        np.random.shuffle(splitted)
        concat = np.concatenate(splitted, axis=0)
        X_transformed[i] = concat
    return X_transformed

# 8. Time Warp (low cost)
def time_warp_transform_low_cost(X: np.ndarray, sigma: float = 0.2, num_knots: int = 4, num_splines: int = 150) -> np.ndarray:
    """
    Stretching and warping the time-series (low cost)
    """
    time_stamps = np.arange(X.shape[1])
    knot_xs = np.arange(0, num_knots + 2, dtype=float) * (X.shape[1] - 1) / (num_knots + 1)
    spline_ys = np.random.normal(loc=1.0, scale=sigma, size=(num_splines, num_knots + 2))

    spline_values = np.array([get_cubic_spline_interpolation(time_stamps, knot_xs, spline_ys_individual) for spline_ys_individual in spline_ys])

    cumulative_sum = np.cumsum(spline_values, axis=1)
    distorted_time_stamps_all = cumulative_sum / cumulative_sum[:, -1][:, np.newaxis] * (X.shape[1] - 1)

    random_indices = np.random.randint(num_splines, size=(X.shape[0] * X.shape[2]))

    X_transformed = np.empty(shape=X.shape)
    for i, random_index in enumerate(random_indices):
        X_transformed[i // X.shape[2], :, i % X.shape[2]] = np.interp(time_stamps, distorted_time_stamps_all[random_index], X[i // X.shape[2], :, i % X.shape[2]])
    return X_transformed

def get_cubic_spline_interpolation(x_eval: np.ndarray, x_data: np.ndarray, y_data: np.ndarray) -> np.ndarray:
    """
    Get values for the cubic spline interpolation
    """
    cubic_spline = CubicSpline(x_data, y_data)
    return cubic_spline(x_eval)

# 9. Time Warp (high cost) - not used
def time_warp_transform_improved(X: np.ndarray, sigma: float = 0.2, num_knots: int = 4) -> np.ndarray:
    """
    Stretching and warping the time-series
    """
    time_stamps = np.arange(X.shape[1])
    knot_xs = np.arange(0, num_knots + 2, dtype=float) * (X.shape[1] - 1) / (num_knots + 1)
    spline_ys = np.random.normal(loc=1.0, scale=sigma, size=(X.shape[0] * X.shape[2], num_knots + 2))

    spline_values = np.array([get_cubic_spline_interpolation(time_stamps, knot_xs, spline_ys_individual) for spline_ys_individual in spline_ys])

    cumulative_sum = np.cumsum(spline_values, axis=1)
    distorted_time_stamps_all = cumulative_sum / cumulative_sum[:, -1][:, np.newaxis] * (X.shape[1] - 1)

    X_transformed = np.empty(shape=X.shape)
    for i, distorted_time_stamps in enumerate(distorted_time_stamps_all):
        X_transformed[i // X.shape[2], :, i % X.shape[2]] = np.interp(time_stamps, distorted_time_stamps, X[i // X.shape[2], :, i % X.shape[2]])
    return X_transformed
