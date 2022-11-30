import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# Convert the patch into a one-dimensional vector
def patch_to_vector(patch_array):
    h = patch_array.shape[0]
    return patch_array.reshape(h*h,3)


# Convert the vector back to patch
def vector_to_patch(vector, h):
    return vector.reshape(h,h,3)


# Channel restoration of the standardized image for easy display
def channel_recover(img_array):
    return np.where(img_array==[-100,-100,-100],img_array,img_array*256).astype(np.int32)