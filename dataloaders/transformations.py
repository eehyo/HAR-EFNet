import numpy as np
import math
from scipy.interpolate import CubicSpline     # for warping
import itertools
from typing import List, Callable, Tuple, Union


# 1. Noise - mtl, simclr
# Random Gaussian noise added to each sample
def noise_transform(X: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    """
    Adding random Gaussian noise with mean 0
    """
    noise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X + noise

# 2. Scaling - mtl, simclr
def scaling_transform(X: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """
    Scaling by a random factor
    """
    scaling_factor = np.random.normal(loc=1.0, scale=sigma, size=(X.shape[0], 1, X.shape[2]))
    return X * scaling_factor

# 3. Rotation - mtl, simclr
def rotation_transform(X: np.ndarray) -> np.ndarray:
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

# 4. Negation - mtl
def negate_transform(X: np.ndarray) -> np.ndarray:
    """
    Inverting the signals
    """
    return X * -1

# 4-1. Negation (probabilistic) - simclr
def negate_transform_probabilistic(X: np.ndarray, p: float = 0.5) -> np.ndarray:
    """
    With probability p, invert the signals; otherwise return X unchanged.
    """
    if np.random.rand() < p:
        return -X
    return X

# # 5. Time Flip (not used)
# def time_flip_transform(X: np.ndarray) -> np.ndarray:
#     """
#     Reversing the direction of time
#     """
#     return X[:, ::-1, :]

# 5-1. Time Flip (probabilistic) - simclr
def time_flip_transform_probabilistic(X: np.ndarray, p: float = 0.5) -> np.ndarray:
    """
    With probability p, reverse the time dimension; otherwise return X unchanged.
    """
    if np.random.rand() < p:
        return X[:, ::-1, :]
    return X


# 6. Channel Shuffle - mtl, simclr
def channel_shuffle_transform(X: np.ndarray) -> np.ndarray:
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

# 7. Time Segment Permutation - mtl, simclr
def time_segment_permutation_transform_improved(X: np.ndarray, nPerm=4, minSegLength=10, max_tries=100) -> np.ndarray:
    """
    Randomly scrambling sections of the signal
    """
    B, T, C = X.shape
    X_new = np.zeros_like(X)

    for b in range(B):
        for _ in range(max_tries):
            segs = np.zeros(nPerm + 1, dtype=int)
            segs[1:-1] = np.sort(np.random.randint(minSegLength, T - minSegLength, nPerm - 1))
            segs[-1] = T

            # Check if all segments are at least minSegLength
            if np.min(segs[1:] - segs[:-1]) > minSegLength:
                break
        else:
            segs = np.linspace(0, T, nPerm + 1, dtype=int)

        idx = np.random.permutation(nPerm)
        pp = 0
        for i in range(nPerm):
            part = X[b, segs[idx[i]]:segs[idx[i] + 1], :]
            X_new[b, pp:pp + len(part), :] = part
            pp += len(part)

    return X_new

# 8. Time Warp (low cost) - simclr
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

# 9. Time Warp (high cost) - simclr(not used)
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

# 10. Time Warping - mtl
# Warps time steps using smoothly varying curves
def time_warp_transform(X, sigma=0.2, knot=4):
    B, T, C = X.shape
    X_new = np.zeros_like(X)
    x_range = np.arange(T)

    for b in range(B):
        tt_new = DistortTimesteps(X[b], sigma=sigma, knot=knot)
        for i in range(C):
            X_new[b, :, i] = np.interp(x_range, tt_new[:, i], X[b, :, i])

    return X_new

def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = np.linspace(0, X.shape[0] - 1, num=knot + 2) 
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, X.shape[1])) 
    x_range = np.arange(X.shape[0])
    warped = np.zeros_like(X)
    for i in range(X.shape[1]):
        cs = CubicSpline(xx, yy[:, i])
        warped[:, i] = cs(x_range)
    return warped

def DistortTimesteps(X, sigma=0.2, knot=4):
    tt = GenerateRandomCurves(X, sigma=sigma, knot=knot)
    tt_cum = np.cumsum(tt, axis=0)  # Add intervals to make a cumulative graph
    t_scale = [(X.shape[0] - 1) / tt_cum[-1, i] for i in range(X.shape[1])]
    for i in range(X.shape[1]):
        tt_cum[:, i] *= t_scale[i]
    return tt_cum


# 11. Horizontally Flipped - mtl
def horizontal_flip_transform(X: np.ndarray) -> np.ndarray: 
    """
    Flips the signal in time (reverses along time axis).
    Input:
        X: [T, C]
    Output:
        X_flipped: [T, C]
    """
    return np.flip(X, axis=0)


# Common transform functions
# noise_transform
# scaling_transform
# rotation_transform
# channel_shuffle_transform
# time_segment_permutation_transform_improved

# MTL-specific transform functions
# negate_transform
# time_warp_transform
# horizontal_flip_transform

# SimCLR-specific transform functions
# negate_transform_probabilistic
# time_flip_transform_probabilistic
# time_warp_transform_low_cost
# time_warp_transform_improved (not used)