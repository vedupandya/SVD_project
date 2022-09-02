# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 00:23:08 2022

@author: Vedant Pandya
"""

# Imports

import numpy as np
from scipy.linalg import svd

"""
Singular Value Decomposition
"""
# define a matrix
X = np.array([[3, 3, 2], [2,3,-2]])
print(X)
# perform SVD
U, singular, V_transpose = svd(X)
# print different components
print("U: ",U)
print("Singular array",singular)
print("V^{T}",V_transpose)

"""
Calculate Pseudo inverse
"""
# inverse of singular matrix is just the reciprocal of each element
singular_inv = 1.0 / singular
# create m x n matrix of zeroes and put singular values in it
s_inv = np.zeros(X.shape)
s_inv[0][0]= singular_inv[0]
s_inv[1][1] =singular_inv[1]
# calculate pseudoinverse
M = np.dot(np.dot(V_transpose.T,s_inv.T),U.T)
print(M)

"""
SVD on image compression
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2gray
import tkinter
from tkinter import filedialog
root = tkinter.Tk()
root.title("Photo Editor")
root.iconify()

file = filedialog.askopenfilename(title='open')
img = Image.open(file)
image = np.array(img)
# convert to grayscale
gray_image = rgb2gray(image)

# calculate the SVD and plot the image
U,S,V_T = svd(gray_image, full_matrices=False)
S = np.diag(S)
fig, ax = plt.subplots(5, 2, figsize=(8, 12))

curr_fig=0
for r in [5, 10, 30, 65, 100]:
    cat_approx =U[:, :r] @ S[0:r, :r] @ V_T[:r, :]
    ax[curr_fig][0].imshow(cat_approx, cmap=plt.cm.gray)
    ax[curr_fig][0].set_title("k = "+str(r))
    ax[curr_fig,0].axis('off')
    ax[curr_fig][1].set_title("Original Image")
    ax[curr_fig][1].imshow(gray_image, cmap=plt.cm.gray)
    ax[curr_fig,1].axis('off')
    curr_fig +=1
plt.show()
root.mainloop()