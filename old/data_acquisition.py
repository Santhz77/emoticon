'''
This takes the bitalino openRevoution sensor ECG data froma txt file
And extracts the basic features from the ECG.
Author : Santhosh Nayak (santhoshnayak0903@gmail.com)
date : 16 -02- 2018
'''

import numpy as np
import matplotlib.pyplot as plt
import biosppy as bs

#load the data fromthe file
x = np.loadtxt('data/santhosh_vr_experiment.txt')[:,-2]


# we need to covert the binary data to original data in mv (milli volts)
# Vcc = 3.7  battery voltage = 3.7 V | Sensor_gain = 1100
# ECG (V)  =  {(data from the bitalino) * Vcc / (2^10) -  1 } -  { Vcc / 2} / { Sensor_gain }
# ECG (mV) = ECG (V) * 1000
# Reference : http://forum.bitalino.com/viewtopic.php?f=12&t=128

# ecg in millivolts
ecg_value_millivolts = []
for i in range(0,len(x)):
    ecg_value_millivolts.append(x[i] * 3.7 / 1023. - (1.5) / 1.1)


# just to plot the signal
# plt.plot(ecg_value_millivolts[-12000:-5000], 'k')
# plt.ylabel('mV')
# plt.xlabel('t (ms)')
# plt.show()
#plt.savefig('SampleECG.png',dpi=300)


print(x)


# data processing
# For more information refer : http://biosppy.readthedocs.io/en/stable/biosppy.signals.html#biosppy-signals-ecg
detected_value = bs.ecg.ecg(signal=x, sampling_rate=100.0, show=True)

