'''
This takes the bitalino openRevoution sensor EDA data froma txt file
And extracts the basic features from the EDA.
Author : Santhosh Nayak (santhoshnayak0903@gmail.com) \ Gautam Sawala
date : 16 -02- 2018
'''

import numpy as np
import matplotlib.pyplot as plt
import biosppy as bs

#load the data fromthe file
x = np.loadtxt('data/sledger_man_eda.txt')[:,-1]
y = np.loadtxt('data/sledger_man_eda.txt')[:,-7]


# Vcc =  battery voltage = 3.7 V | Sensor_gain = 1100
# RMOhm = 1 - EDAB / 2^n (sensor resistance in mega ohms)
# EDAS = 1 / RMOhm (conductance in microsiemens)
# Reference : http://forum.bitalino.com/viewtopic.php?f=12&t=128

# ecg in muesiemens
eda_value_microsiemens = []
for i in range(0,len(x)):
    eda_value_microsiemens.append(1*1000/ (1- (x[i]/1023))) 

y_eda_value_microsiemens = []
for j in range(0,len(y)):
    r = 1- (y[j]/1023)
    eda_mSiemens = 1/r
    y_eda_value_microsiemens.append(eda_mSiemens)

print(y_eda_value_microsiemens )



# just to plot the signal
# plt.plot(eda_value_microsiemens, 'k')
plt.plot(y_eda_value_microsiemens[1000:10000], 'k')
plt.ylabel('microSiemens')
plt.xlabel('t (ms)')
plt.show()
#plt.savefig('SampleECG.png',dpi=300)


# data processing
# For more information refer : http://biosppy.readthedocs.io/en/stable/biosppy.signals.html
#detected_value = bs.eda.eda(signal=eda_value_muesiemens, sampling_rate=10.0, show=True, min_amplitude=0.1)
#detected_value =bs.eda.eda(signal = eda_value_microsiemens, sampling_rate = 10, show = True)