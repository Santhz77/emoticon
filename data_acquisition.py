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
#x = np.loadtxt('raw_data/subject_3_2018-03-20 18:31:37.740803_vr_experience.txt')[:,-1]
cavfile = "raw_data/subject_3_2018-03-20 18:31:37.740803_vr_experience.txt"

eda_raw = []
with open(cavfile, 'r') as myfile:
    data=myfile.readlines()
    for line in data:
        floats = [float(x) for x in line.split(', ')]
        eda_raw.append(floats[5])



# we need to covert the binary data to original data in mv (milli volts)
# Vcc = 3.7  battery voltage = 3.7 V | Sensor_gain = 1100
# ECG (V)  =  {(data from the bitalino) * Vcc / (2^10) -  1 } -  { Vcc / 2} / { Sensor_gain }
# ECG (mV) = ECG (V) * 1000
# Reference : http://forum.bitalino.com/viewtopic.php?f=12&t=128

# ecg in millivolts
def eda_bin_to_microsiemens(eda):
    '''
    Vcc =  battery voltage = 3.7 V | Sensor_gain = 1100
    RMOhm = 1 - EDAB / 2^n (sensor resistance in mega ohms)
    EDAS = 1 / RMOhm (conductance in microsiemens)
    Reference : http://forum.bitalino.com/viewtopic.php?f=12&t=128

    :param eda: eda array
    :return:
    '''
    # convert binary data to micro siemens
    eda_value_microsiemens = []
    for j in range(0, len(eda)):
        r = 1 - (eda[j] / 1023)
        eda_mSiemens = 1 / r
        eda_value_microsiemens.append(eda_mSiemens)

    return eda_value_microsiemens
#
# # just to plot the signal
# y = ecg_bin_to_millivolts(x)


eda = eda_bin_to_microsiemens(eda_raw)
print(eda)
print(len(eda))
# data processing
# For more information refer : http://biosppy.readthedocs.io/en/stable/biosppy.signals.html#biosppy-signals-ecg
detected_value = bs.eda.eda(signal=eda[1003], sampling_rate=1000.0, show=True)

print(detected_value['filtered'])

