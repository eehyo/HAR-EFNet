from scipy.interpolate import CubicSpline      # for warping
from transforms3d.axangles import axangle2mat  # for rotation
import numpy as np


"""
Based on work of T. T. Um et al.: https://arxiv.org/abs/1706.00527
Contact: Terry Taewoong Um (terry.t.um@gmail.com)

T. T. Um et al., “Data augmentation of wearable sensor data for parkinson’s 
disease monitoring using convolutional neural networks,” in Proceedings of 
the 19th ACM International Conference on Multimodal Interaction, ser. ICMI 
2017. New York, NY, USA: ACM, 2017, pp. 216–220.

https://dl.acm.org/citation.cfm?id=3136817

@inproceedings{TerryUm_ICMI2017, author = {Um, Terry T. and Pfister, Franz M. 
J. and Pichler, Daniel and Endo, Satoshi and Lang, Muriel and Hirche, Sandra 
and Fietzek, Urban and Kuli\'{c}, Dana}, title = {Data Augmentation of 
Wearable Sensor Data for Parkinson's Disease Monitoring Using Convolutional 
Neural Networks}, booktitle = {Proceedings of the 19th ACM International 
Conference on Multimodal Interaction}, series = {ICMI 2017}, year = {2017}, 
isbn = {978-1-4503-5543-8}, location = {Glasgow, UK}, pages = {216--220}, 
numpages = {5}, doi = {10.1145/3136755.3136817}, acmid = {3136817}, publisher 
= {ACM}, address = {New York, NY, USA}, keywords = {Parkinson\&#39;s disease, 
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
    xx = np.linspace(0, X.shape[0] - 1, num=knot + 2) # 모든 채널이 공유하는 시간 포인트
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, X.shape[1])) # 채널별 다른 랜덤 값
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

# 5. Rotation
def DA_Rotation(X):
    axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(X , axangle2mat(axis,angle))

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
# Randomly crops a subwindow and resizes back to original length
def DA_Cropping(X, crop_ratio=0.9):
    """
    X: np.ndarray of shape [T, C]
    crop_ratio: proportion of original length to retain (e.g., 0.9 means 90% kept)
    """
    T, C = X.shape
    crop_len = int(T * crop_ratio)
    start = np.random.randint(0, T - crop_len)
    cropped = X[start:start + crop_len]

    # Resize to original length using linear interpolation
    resized = np.zeros((T, C))
    for i in range(C):
        resized[:, i] = np.interp(
            np.linspace(0, crop_len - 1, num=T),
            np.arange(crop_len),
            cropped[:, i]
        )
    return resized