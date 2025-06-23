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
    scaling_factor = np.random.normal(loc=1.0, scale=sigma, size=(X.shape[0], 1, X.shape[2]))
    return X * scaling_factor

# 3. Rotation for 9-axis IMU data (Sensor-wise Rotation)
def DA_Rotation_per_sensor(X):
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
    B, T, C = X.shape
    X_new = np.zeros_like(X)
    for b in range(B):
        while True:
            segs = np.zeros(nPerm + 1, dtype=int)
            segs[1:-1] = np.sort(np.random.randint(minSegLength, T - minSegLength, nPerm - 1))
            segs[-1] = T
            if np.min(segs[1:] - segs[:-1]) > minSegLength:
                break
        idx = np.random.permutation(nPerm)
        pp = 0
        for i in range(nPerm):
            part = X[b, segs[idx[i]]:segs[idx[i]+1], :]
            X_new[b, pp:pp+len(part), :] = part
            pp += len(part)
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
    B, T, C = X.shape
    X_new = np.zeros_like(X)
    x_range = np.arange(T)

    for b in range(B):
        tt_new = DistortTimesteps(X[b], sigma=sigma, knot=knot)
        for i in range(C):
            X_new[b, :, i] = np.interp(x_range, tt_new[:, i], X[b, :, i])

    return X_new

# 8. Channel-Shuffled
def DA_ChannelShuffle(X):
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

