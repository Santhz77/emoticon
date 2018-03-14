from local_bitalino import BITalino
import time
import numpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import biosppy as bs # maybe not used!

import scipy.io as sio
from scipy import signal

from socket import socket
import peakutils


def tostring(data):
    """
    :param data: object to be converted into a JSON-compatible `str`
    :type data: any
    :return: JSON-compatible `str` version of `data`

    Converts `data` from its native data type to a JSON-compatible `str`.
    """
    dtype = type(data).__name__
    if dtype == 'ndarray':
        if numpy.shape(data) != ():
            data = data.tolist()  # data=list(data)
        else:
            data = '"' + data.tostring() + '"'
    elif dtype == 'dict' or dtype == 'tuple':
        try:
            data = json.dumps(data)
        except:
            pass
    elif dtype == 'NoneType':
        data = ''
    elif dtype == 'str' or dtype == 'unicode':
        data = json.dumps(data)

    return str(data)



hostname = 'localhost'
port_number = 7000
def send_to_server(data_as_json):
    #instanciate a socket
    sock = socket()
    sock.connect((hostname, 7000))

    sock.send(data_as_json.encode('utf-8'))
    print('ecg data sent . ')
    sock.close()


labels = ["'nSeq'", "'I1'", "'I2'", "'O1'", "'O2'", "'A1'", "'A2'", "'A3'", "'A4'", "'A5'", "'A6'"]

# initial settings
macAddress = '20:16:12:22:01:28'
running_time = 30
batteryThreshold = 30
acqChannels = [1] # for A2 of bitalino
samplingRate = 1000 # 1k Hz.
nSamples = 100 # Number of samples to extract
digitalOutput = [1, 1] # just the output we perform (from machine to device )


Fs = float(samplingRate)
szplot = 500 # to show the plot (show for last)


# lowpass ECG
Wn = 30.0 * 2 / float(Fs);
NN = 3.0;
[ale, ble] = signal.butter(NN, Wn, 'low', analog=False);

# highpass ECG
Wn = 5.0 * 2 / float(Fs);
NN = 3.0;
[ahe, bhe] = signal.butter(NN, Wn, 'high', analog=False);



# Connect to BITalino
device = BITalino(macAddress)
print("device connected to bitalino")


# Set battery threshold
device.battery(batteryThreshold)


#cols = numpy.arange(len(acqChannels) + 5)

# Start Acquisition
device.start(samplingRate, acqChannels)

start = time.time()
end = time.time()


#time initialization
timeend = 0.0
timeinit = 0.0 # initial time
timeend += float(nSamples) / float(samplingRate) # end time ( in our case its 300/1000 = 0.3 sec)
time_elapsed = []

ecg= []
peakind = []

# plotting
fig = plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_subplot(111)
plt.ion()
plt.xlabel('Time (seconds)')
line1, = ax.plot(time_elapsed, ecg, 'b-') # raw data
line2, = ax.plot(time_elapsed, ecg, 'r-') # to represent teh filtered data
line3, = ax.plot(time_elapsed,peakind, 'g*')
fig.show()
fig.canvas.draw()


data = []

try:
    #indefinite signal capture
    while 1:

        # read data from the device
        received_data = device.read(nSamples)

        data = np.concatenate((data,received_data[:,-1]),axis = 0)

        # we detrend the data
        ecg = signal.detrend(data)

        #ecg = data



        # #bandpass filtering the data
        ecg_filtered = signal.filtfilt(ale, ble, ecg); # lowpass filter
        ecg_filtered = signal.filtfilt(ahe, bhe, ecg_filtered); # high pass filter

        peakind = peakutils.indexes(ecg, thres=0.02 / max(ecg), min_dist=1)
        peak_time = []
        peak_val = []




        #update time
        time_elapsed = np.concatenate((time_elapsed, np.linspace(timeinit, timeend, nSamples + 1)[1:]), 0)
        timeinit = time_elapsed[-1]
        timeend += float(nSamples) / float(samplingRate)

        # for k in peakind:
        #     peak_time.append(time_elapsed[k])
        #     peak_val.append(ecg[k])
        #
        #
        # print(len(peak_time[-szplot:]))
        # print(len(peak_val[-szplot:]))
        # line3.set_ydata(peak_val[-szplot:])  # / max(ecg_filtered[-szplot:]
        # line3.set_xdata(peak_time[-szplot:])

        # update plot
        line1.set_ydata(ecg[-szplot:]) # / max(ecg[-szplot:])
        line1.set_xdata(time_elapsed[-szplot:])

        line2.set_ydata(ecg_filtered[-szplot:] ) #/ max(ecg_filtered[-szplot:]
        line2.set_xdata(time_elapsed[-szplot:])

        ax.relim()
        ax.autoscale_view()

        fig.canvas.draw()
        plt.draw()


        #send data to server as a json
        # { "ecg" : "[data]" }
        data_as_json = "{ \"ecg\" : "
        data_as_json = data_as_json + tostring(ecg[-szplot:]) + '}'
        print(data_as_json)
        send_to_server(data_as_json)


except KeyboardInterrupt:
    print("Keyboard interupted")

    # Turn BITalino led on
    device.trigger(digitalOutput)
    # Stop acquisition
    device.stop()
    # Close connection
    device.close()





# while (end - start) < running_time:
#     # Read samples
#     # print(device.read(nSamples))
#
#
#
#
#     # print(ECG)
#     # envelope = numpy.mean(abs(numpy.diff(ECG)))
#     # print(envelope)
#
#     # res = "{"
#     # for i in cols:
#     #     idx = i
#     #     if (i > 4): idx = acqChannels[i - 5] + 5
#     #     res += '"' + labels[idx] + '":' + tostring(data[:, i]) + ','
#     # res = res[:-1] + "}"
#
#     # print (res)
#
#     # we convert the collected samples to millivolts
#     # ecg_value_millivolts = []
#     # for i in range(0, len(ECG)):
#     #     ecg_value_millivolts.append(ECG[i] * 3.7 / 1023. - (1.5) / 1.1)
#
#     # detected_value = bs.ecg.ecg(signal=ecg_value_millivolts, sampling_rate=1000.0, show=False)
#
#     # ecg = bs.ecg()
#     #
#     # filtered_data = bs.ecg.
#
#
#     # ax.clear()
#
#
#
#     ax.plot(ecg_value_millivolts)
#     # ax.plot(detected_value['filtered'])
#     fig.canvas.draw()
#
#     print("============================")
#
#     end = time.time()
#
#
