# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 10:47:11 2026

@author: osips
"""

import numpy as np
import matplotlib.pyplot as plt

#%% exit angle aspheric 
#Flat left side, convex right side
#Point source on the left, collimating on the right
#

sin_alpha=np.linspace(0, 0.5, 100)

n1=1.8
n2=1.45

sin_beta_1 = sin_alpha / np.sqrt(n1 ** 2 + 1 - 2 * np.sqrt(n1 ** 2 - sin_alpha ** 2))
sin_beta_2 = sin_alpha / np.sqrt(n2 ** 2 + 1 - 2 * np.sqrt(n2 ** 2 - sin_alpha ** 2))

plt.figure()
plt.plot(sin_alpha, 180 / np.pi * np.arcsin(sin_beta_1), sin_alpha, 180 / np.pi * np.arcsin(sin_beta_2))
plt.plot(sin_alpha, 180 / np.pi * np.arcsin(sin_alpha), '--r')
plt.grid()
plt.ylim(0,60)
plt.xlim(0,0.5)
#plt.legend(['Collimated side n='+str(n1), 'Collimated side n='+str(n2), 'focused side, arcsin(NA)'])
plt.xlabel('NA')
plt.ylabel('Angle of incidence')
plt.title('Angles of incidence in an aspheric lens')

#%% Aspheric with equal angles of incidence
alpha=np.linspace(0, np.pi / 4, 200)# angle of incidence

beta1 = np.arcsin(np.sin(alpha)/n1)
beta2 = np.arcsin(np.sin(alpha)/n2)
theta1 = alpha-beta1
theta2 = alpha-beta2
#plt.figure()
#plt.plot(2*180/np.pi*theta1, 180/np.pi*alpha)
#plt.plot(2*180/np.pi*theta2, 180/np.pi*alpha)

plt.plot(np.sin(2*theta1), 180/np.pi*alpha, 'blue') #x axis is focused side NA
plt.plot(np.sin(2*theta2), 180/np.pi*alpha, 'red')

plt.legend(['|) collimated side n='+str(n1), '|) collimated side n='+str(n2), '|) focused side, arcsin(NA)', 'equal angles n='+str(n1), 'equal angles n='+str(n2)])
plt.show()
