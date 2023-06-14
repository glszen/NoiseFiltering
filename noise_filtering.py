# -*- coding: utf-8 -*-
"""
**********Emotions Data Analysis**********
**********@author: gulsen************
"""
""" *********************************************************************** """
""" 
Libraries
"""

import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('data_science/Noise_Filtering'):
    for filename in filenames:
        print(os.path.join(dirname,filename))

import cv2
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error
from skimage.restoration import denoise_bilateral

import plotly.graph_objects as go

from skimage.util import random_noise

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

""" *********************************************************************** """
"""
Data Import
"""

img_test= cv2.imread(r'C:\Users\LENOVO\Desktop\data_science\Noise_Filtering\noised.png', 0)

img_test.shape

""" *********************************************************************** """

"""
Data PROCESSİNG
"""

plt.figure(figsize=(10,10))
plt.imshow(img_test, cmap='gray')

fig=go.Figure(data=[go.Surface(z=img_test, colorscale='gray', colorbar=None)])

fig.update_layout(title='Original Image', autosize=False,
                  width=800, height=800,
                  margin=dict(l=65, r=50, b=65))
fig.update_traces(showscale=False)

fig.show()


noise_factor=10 #adding noise in data

img_test_noisy=img_test

# ZNCC Evalaute:
    
img_test_noisy_zncc = img_test + 50 * np.random.normal(loc=0.0, scale=1.0, size=img_test.shape) #loc: mean of the dist. scale: SD of the dist.

zncc_1image = np.corrcoef(img_test.ravel(), img_test_noisy_zncc.ravel())[1,0]
psnr_1image = peak_signal_noise_ratio(img_test, img_test_noisy_zncc)
ssim_1image = structural_similarity(img_test, img_test_noisy_zncc, channel_axis=False)

plt.figure(figsize=(10,10))

plt.subplot(1,2,1), plt.imshow(img_test, cmap='gray'), plt.title('Original',)
plt.xticks([]), plt.yticks([])

plt.subplot(1,2,2), plt.imshow(img_test_noisy_zncc, cmap='gray'), plt.title('Noised Image')

plt.suptitle('After Noise Factor:50 => PSNR:%0.4f ZNCC:%0.4f SSIM:%0.4f' %(psnr_1image, zncc_1image, ssim_1image),y=0.25)
plt.xticks([]), plt.yticks([])

img_test_noisy = img_test_noisy.astype('float32')

plt.figure(figsize=(10,10)) # Displaying the image
plt.imshow(img_test_noisy, cmap='gray')

plt.savefig('noised.png')

# Side by Side Comparision

plt.figure(figsize=(20,20))

plt.subplot(1,2,1),plt.imshow(img_test, cmap='gray'), plt.title('Original',)
plt.xticks([]), plt.yticks([])

plt.subplot(1,2,2),plt.imshow(img_test_noisy, cmap='gray'), plt.title('Noised Image',)
plt.xticks([]), plt.yticks([])

# Using Mean Filter to Denoise Image:

kernel = np.ones((3,3), np.float32) / 9

denoised_mean = cv2.filter2D(img_test_noisy, -1, kernel) #-1:Depth of the output image

def function_image(img_test, img_test_noisy, denoised, filter_type):
    
    plt.figure(figsize=(20,20))
    
    plt.subplot(1,3,1), plt.imshow(img_test, cmap='gray'), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1,3,2), plt.imshow(img_test_noisy, cmap='gray'), plt.title('Noised Image')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1,3,3), plt.imshow(denoised, cmap='gray'), plt.title(filter_type)
    plt.xticks([]), plt.yticks([])
    plt.show()
    
    plt.savefig(str(filter_type)+'.png')
    
function_image(img_test, img_test_noisy, denoised_mean, 'Mean Filter')

#Evaluating Images:

def evaluate_image(filtername, img_test, img_test_noisy, denoised):
    
    psnr_orgnoised=peak_signal_noise_ratio(img_test, img_test_noisy)
    psnr_orgdenoised=peak_signal_noise_ratio(img_test, denoised)
    
    ssim_orgnoised=structural_similarity(img_test, img_test_noisy, chanel_axis=False) #SSIM is used for measuring the similarity between two images.
    ssim_orgdenoised=structural_similarity(img_test, denoised, chanel_axis=False)
    
    data=[{'Filter Name': filtername, 'PSNR ORG-NOISED' :psnr_orgnoised, 'PSNR ORG-DENOISED':psnr_orgdenoised, 'SSIM ORG-NOISED':ssim_orgnoised, 'SSIM ORG-DENOISED':ssim_orgdenoised}]
    df=pd.DataFrame(data)
    
    return df

#Using Gaussian Filter to Denoise Image

denoised_gaussian_05 = cv2.GaussianBlur(img_test_noisy, (3,3), 0.5)

function_image(img_test, img_test_noisy, denoised_gaussian_05, 'Gaussian Filter')

denoised_gaussian_05_df=evaluate_image(0.5, img_test, img_test_noisy, denoised_gaussian_05)
denoised_gaussian_05_df

#Larger Sigma Value 1:
    
denoised_gaussian_2=cv2.GaussianBlur(img_test_noisy, (3,3), 2)
denoised_gaussian_2_df=evaluate_image(2, img_test, img_test_noisy, denoised_gaussian_2)
denoised_gaussian_2_df

#Larger Sigma Value 2:
    
denoised_gaussian_5=cv2.GaussianBlur(img_test_noisy, (3,3), 5)
denoised_gaussian_5_df=evaluate_image(5, img_test, img_test_noisy, denoised_gaussian_5)
denoised_gaussian_5_df

#Overall Comparision:
    
gaussian_df_final= pd.concat([denoised_gaussian_05_df, denoised_gaussian_2_df, denoised_gaussian_5_df])

gaussian_df_final=gaussian_df_final.rename(columns={'Filter Name':'Gaussian Value'})
gaussian_df_final

fig, (ax1,ax2)=plt.subplots(1,2,tight_layout=True) #tight_layout:subplot(s) fits in to the figure area
fig.suptitle('Change in PSNR and SSIM with change in Standart Deviation in Gaussian Filter')

ax1.plot(gaussian_df_final['Gaussian Value'], gaussian_df_final['PSNR ORG-DENOISED'], 'b-o', label = "PSNR") #b-o:a chatbot written
ax1.set(xlabel='Gaussian Value', ylabel='PSNR Value')

ax2.plot(gaussian_df_final['Gaussian Value'], gaussian_df_final['SSIM ORG-DENOISED'],'b-o', label="SSIM")
ax2.set(xlabel='Gaussian Value', ylabel='SSIM Value')

#Using Median Filter to Denoise Image:
    
denoised_median= cv2.medianBlur(img_test_noisy.astype('float32'),3,3)

function_image(img_test, img_test_noisy, denoised_median, 'Median Filter')

denoised_median_df= evaluate_image('Gaussian', img_test, img_test_noisy, denoised_median)

#Using Bilateral Filter to Denoise Image:
    
denoised_bilateral_3311=cv2.bilateralFilter(img_test_noisy, 9,1,1)

function_image(img_test, img_test_noisy, denoised_bilateral_3311, 'Bilateral Filter')

denoised_bilateral_3311_df=evaluate_image('Bilateral', img_test, img_test_noisy, denoised_bilateral_3311)
denoised_bilateral_3311_df

#Concatening Gaussian, Median and Bilateral Filter:
    
compare_df_final=pd.concat([denoised_gaussian_05_df, denoised_median_df, denoised_bilateral_3311_df])
compare_df_final['Filter Name']=compare_df_final['Filter Name'].replace([0.5,'Gaussian','Bilateral'], ['Gaussian', 'Median', 'Bilateral'])
compare_df_final