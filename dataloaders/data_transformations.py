from scipy.interpolate import CubicSpline      # for warping
import math
import numpy as np


"""
Based on work of T. T. Um et al.: https://arxiv.org/abs/1706.00527
Contact: Terry Taewoong Um (terry.t.um@gmail.com)

T. T. Um et al., "Data augmentation of wearable sensor data for parkinson's 
disease monitoring using convolutional neural networks," in Proceedings of 
the 19th ACM International Conference on Multimodal Interaction, ser. ICMI 
2017. New York, NY, USA: ACM, 2017, pp. 216-220.

https://dl.acm.org/citation.cfm?id=3136817

@inproceedings{TerryUm_ICMI2017, author = {Um, Terry T. and Pfister, Franz M. 
J. and Pichler, Daniel and Endo, Satoshi and Lang, Muriel and Hirche, Sandra 
and Fietzek, Urban and Kuli\'{c}, Dana}, title = {Data Augmentation of 
Wearable Sensor Data for Parkinson's Disease Monitoring Using Convolutional 
Neural Networks}, booktitle = {Proceedings of the 19th ACM International 
Conference on Multimodal Interaction}, series = {ICMI 2017}, year = {2017}, 
isbn = {978-1-4503-5543-8}, location = {Glasgow, UK}, pages = {216--220}, 
numpages = {5}, doi = {10.1145/3136755.3136817}, acmid = {3136817}, publisher 
= {ACM}, address = {New York, NY, USA}, keywords = {Parkinson&#39;s disease, 
convolutional neural networks, data augmentation, health monitoring, 
motor state detection, wearable sensor}, }
"""



# 1. Jittering
# Random Gaussian noise added to each sample
def DA_Jitter(X, sigma=0.05):
    noise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X + noise

# 2. Scaling
# Random scaling factor applied per channel
def DA_Scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
    scalingNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * scalingNoise

# 3. Magnitude Warping
# Smoothly varying curve applied as multiplicative noise
## This example using cubic splice is not the best approach to generate random curves. 
## You can use other aprroaches, e.g., Gaussian process regression, Bezier curve, etc.
def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = np.linspace(0, X.shape[0] - 1, num=knot + 2) 
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, X.shape[1])) 
    x_range = np.arange(X.shape[0])
    warped = np.zeros_like(X)
    for i in range(X.shape[1]):
        cs = CubicSpline(xx, yy[:, i])
        warped[:, i] = cs(x_range)
    return warped

def DA_MagnitudeWarp(X, sigma=0.2, knot=4):
    curves = GenerateRandomCurves(X, sigma=sigma, knot=knot)
    return X * curves

# 4. Time Warping
# Warps time steps using smoothly varying curves
def DistortTimesteps(X, sigma=0.2, knot=4):
    tt = GenerateRandomCurves(X, sigma=sigma, knot=knot)
    tt_cum = np.cumsum(tt, axis=0)  # Add intervals to make a cumulative graph
    t_scale = [(X.shape[0] - 1) / tt_cum[-1, i] for i in range(X.shape[1])]
    for i in range(X.shape[1]):
        tt_cum[:, i] *= t_scale[i]
    return tt_cum

def DA_TimeWarp(X, sigma=0.2, knot=4):
    tt_new = DistortTimesteps(X, sigma=sigma, knot=knot)
    x_range = np.arange(X.shape[0])
    X_new = np.zeros_like(X)
    for i in range(X.shape[1]):
        X_new[:, i] = np.interp(x_range, tt_new[:, i], X[:, i])
    return X_new


# 5-2. Rotation for 9-axis IMU data (Sensor-wise Rotation)
def DA_Rotation_per_sensor(X):
    """
    Applies a different random 3D rotation to each sensor triplet (x, y, z).
    For input [x1, y1, z1, x2, y2, z2, x3, y3, z3], each (x, y, z) group gets its own rotation.

    Input:
        X: [T, 9] - 3 sensors × 3-axis (x, y, z)
    Output:
        X_rot: [T, 9] - rotated data with different R per sensor
    """
    assert X.shape[1] == 9, f"Expected 9 channels but got {X.shape[1]}"
    X_rot = np.zeros_like(X)

    for i in range(3):  # For each of the 3 sensor groups
        axis = np.random.uniform(low=-1, high=1, size=3)
        axis = axis / np.linalg.norm(axis)
        angle = np.random.uniform(low=-np.pi, high=np.pi)
        R = axangle2mat(axis, angle, is_normalized=True)  # 개별 회전 행렬

        vec = X[:, i*3:(i+1)*3]  # [T, 3]
        X_rot[:, i*3:(i+1)*3] = vec @ R  # 회전 적용

    return X_rot

def axangle2mat(axis, angle, is_normalized=False):
    ''' Rotation matrix for rotation angle `angle` around `axis`
    Notes
    -----
    Reference: the Transforms3d package - transforms3d.axangles.axangle2mat
    From: http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    '''
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

# 6. Permutation
# Splits the signal into segments and randomly permutes them
def DA_Permutation(X, nPerm=4, minSegLength=10):
    X_new = np.zeros_like(X)
    while True:
        segs = np.zeros(nPerm + 1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0] - minSegLength, nPerm - 1))
        segs[-1] = X.shape[0]
        if np.min(segs[1:] - segs[:-1]) > minSegLength:
            break
    idx = np.random.permutation(nPerm)
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]]:segs[idx[ii] + 1], :]
        X_new[pp:pp + len(x_temp), :] = x_temp
        pp += len(x_temp)
    return X_new

# 7. Cropping
# Randomly crops a contiguous window from the time series
def DA_Cropping(X, crop_size, start=None):
    """
    Crops a segment of length `crop_size` and zero-pads it to match original length.

    Parameters:
    - X: np.ndarray of shape (T, C)
    - crop_size: int, desired crop length
    - start: int, optional starting index. If None, randomly selected.

    Returns:
    - X_padded: np.ndarray of shape (T, C)
    """
    T, C = X.shape
    assert crop_size <= T, "crop_size must be less than or equal to the sequence length"

    max_start = T - crop_size
    if start is None:
        start = np.random.randint(0, max_start + 1)
    else:
        start = max(0, min(start, max_start))  # ❗ 보정 포인트

    X_crop = X[start:start + crop_size, :]  # [crop_size, C]

    # Zero-pad to original length T
    pad_len = T - crop_size
    X_padded = np.pad(X_crop, pad_width=((0, pad_len), (0, 0)), mode='constant')

    return X_padded  # shape: (T, C)