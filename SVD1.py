# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 23:37:57 2022

@author: Vedant Pandya
"""

import numpy as np
import matplotlib.pyplot as plt
# Load the image
img = plt.imread('noised.jpeg')
# Calculate U (u), Σ (s) and V (vh)
u, s, vh = np.linalg.svd(img, full_matrices=False)
# Remove sigma values below threshold (250)
s_cleaned = np.array([si if si > 250 else 0 for si in s])
# Calculate A' = U * Σ (cleaned) * V
img_denoised = np.array(np.dot(u * s_cleaned, vh), dtype=int)
# Save the new image
plt.imsave('morocco_denoised.jpeg', img_denoised)