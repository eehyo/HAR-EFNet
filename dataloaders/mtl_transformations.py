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

# 3. Rotation for 9-axis IMU data (Sensor-wise Rotation)
def DA_Rotation_per_sensor(X):
    """
    Applies a different random 3D rotation to each sensor triplet (x, y, z).
    For input [x1, y1, z1, x2, y2, z2, x3, y3, z3], each (x, y, z) group gets its own rotation.

    Input:
        X: [T, C] - Channel dimension should be multiple of 3 for sensor axes (x, y, z)
    Output:
        X_rot: [T, C] - rotated data with different R per sensor
    """
    C = X.shape[1]
    if C % 3 != 0:
        raise ValueError(f"Channel dimension ({C}) must be multiple of 3 for proper sensor rotation")
    
    n_sensors = C // 3
    X_rot = np.zeros_like(X)

    for i in range(n_sensors):  # For each sensor group
        axis = np.random.uniform(low=-1, high=1, size=3)
        axis = axis / np.linalg.norm(axis)
        angle = np.random.uniform(low=-np.pi, high=np.pi)
        R = axangle2mat(axis, angle, is_normalized=True)  # Individual rotation matrix

        vec = X[:, i*3:(i+1)*3]  # [T, 3]
        X_rot[:, i*3:(i+1)*3] = vec @ R  # Apply rotation

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

# 4. Negated
# Inverts the sign of the entire signal
def DA_Negated(X):
    """
    Negates (flips sign) of the input signal.
    Input:
        X: [T, C]
    Output:
        -X: [T, C]
    """
    return -X

# 5. Horizontally Flipped
# Reverses the time dimension
def DA_HorizontalFlip(X):
    """
    Flips the signal in time (reverses along time axis).
    Input:
        X: [T, C]
    Output:
        X_flipped: [T, C]
    """
    return np.flip(X, axis=0)


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


# 7. Time Warping
# Warps time steps using smoothly varying curves
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

def DA_TimeWarp(X, sigma=0.2, knot=4):
    tt_new = DistortTimesteps(X, sigma=sigma, knot=knot) 
    x_range = np.arange(X.shape[0]) 
    X_new = np.zeros_like(X)

    for i in range(X.shape[1]):
        X_new[:, i] = np.interp(x_range, tt_new[:, i], X[:, i])
        # sort_idx = np.argsort(tt_new[:, i])
        # sorted_tt = tt_new[sort_idx, i]
        # sorted_X = X[sort_idx, i]
        # X_new[:, i] = np.interp(x_range, sorted_tt, sorted_X)

    return X_new


# 8. Channel-Shuffled
# Shuffles the order of channels (feature dimensions)
def DA_ChannelShuffle(X):
    """
    Randomly shuffles the channels (columns) of the signal.
    Input:
        X: [T, C]
    Output:
        X_shuffled: [T, C]
    """
    C = X.shape[1]
    perm = np.random.permutation(C)
    return X[:, perm]

