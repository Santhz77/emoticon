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
import configparser
import getopt
import sys,os



WEB_HOST_ADDRESS = ""
WEB_PORT = 1234
plot = False

lowcut = 30
highcut = 200

labels = ["'nSeq'", "'I1'", "'I2'", "'O1'", "'O2'", "'A1'", "'A2'", "'A3'", "'A4'", "'A5'", "'A6'"]

# initial settings - default settings
macAddress = '20:16:12:22:01:28'
running_time = 30
batteryThreshold = 30
acqChannels = [1] # 1 for A2
samplingRate = 1000
nSamples = 100
digitalOutput =[1,1]


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

def send_to_server(data_as_json):
    '''
    function to send the data to the websocket.
    :param data_as_json:
    :return:
    '''
    #instanciate a socket
    sock = socket()

    print(WEB_HOST_ADDRESS
          ,WEB_PORT)
    #connect to the socket
    sock.connect((WEB_HOST_ADDRESS,int(WEB_PORT)))

    # send the data as a json
    sock.send(data_as_json.encode('utf-8'))

    # close the connection.
    sock.close()

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def getpeak(ecg, time):
    '''detects the peak from the array!'''
    if(len(ecg) < 500):
        return [0]
    else:
        indexes = peakutils.indexes(ecg, thres=0.8 , min_dist=10)
        return indexes


def get_time_domain_features(ecg,peaklist,fs):
    RR_list = []
    cnt = 0
    print(peaklist)
    while (cnt < (len(peaklist) - 1)):
        RR_interval = (peaklist[cnt + 1] - peaklist[cnt])  # Calculate distance between beats in # of samples
        s_dist = (RR_interval / fs)
        RR_list.append(s_dist)  # Append to list
        cnt += 1

    hr = 60 / np.mean(RR_list)  * 0.1  # 60sec (1 minute) / average R-R interval of signal * our number of samples.
    return hr

def bitalino_data_collection():
    '''
    The core function of the file.
    :return:
    '''

    Fs = float(int(samplingRate))

    szplot = 500  # to show the plot (show for last)

    # Connect to BITalino
    device = BITalino(macAddress)
    print("device connected to bitalino")

    # Set battery threshold
    device.battery(batteryThreshold)

    # cols = numpy.arange(len(acqChannels) + 5)

    # Start Acquisition
    device.start(samplingRate, acqChannels)

    # start = time.time()
    # end = time.time()

    # time initialization
    timeend = 0.0
    timeinit = 0.0  # initial time
    timeend += float(nSamples) / float(samplingRate)  # end time ( in our case its 100/1000 = 0.1 sec)
    time_elapsed = []

    ecg = []
    peakind = []

    # plotting
    if(plot) :
        fig = plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111)
        plt.ion()
        plt.xlabel('Time (seconds)')
        line1, = ax.plot(time_elapsed, ecg, 'b-' , alpha=0.3, label='RAW data')  # raw data
        line2, = ax.plot(time_elapsed, ecg, 'g-', alpha=0.7 ,label='filtered data')  # to represent teh filtered data
        line3, = ax.plot(time_elapsed, ecg, 'ro' , label='detected peak') # peaks
        fig.show()
        fig.canvas.draw()


    data = []

    try:
        # indefinite signal capture
        while 1:
            # read data from the device
            received_data = device.read(nSamples)

            data = np.concatenate((data, received_data[:, -1]), axis=0)

            # we detrend the data
            ecg = signal.detrend(data)


            #bandpassfilter
            ale,ble = butter_bandpass(lowcut, highcut , Fs)
            ecg_filtered = signal.filtfilt(ale, ble, ecg);


            # update time
            time_elapsed = np.concatenate((time_elapsed, np.linspace(timeinit, timeend, nSamples + 1)[1:]), 0)
            timeinit = time_elapsed[-1]
            timeend += float(nSamples) / float(samplingRate)


            # update plot everytime you recive the data
            # note that we show the user past 500 data samples and hence data from past 0.5 second = 500 msec(millisec)
            x = time_elapsed[-szplot:]
            y_raw = ecg[-szplot:]
            y_filtered = ecg_filtered[-szplot:]

            # we now find peaks for past 0.5 seconds ( R peak detection)
            peakind = getpeak(y_filtered, x)
            #  some adjustments to plot the data
            x_peaks = [x[i] for i in peakind]  # peak time
            y_peaks = [y_raw[i] for i in peakind]  # peak value

            heart_rate = get_time_domain_features(y_filtered, peakind,Fs)


            if (plot):
                line1.set_data(x,y_raw)
                line2.set_data(x, y_filtered)
                line3.set_data(x_peaks,y_peaks)

                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                plt.legend(handles=[line1,line2,line3]) # to add the legend.
                plt.draw()

            # send data to server as a json
            # note we  send the last 500
            #{ "ecg" : "[data]" }
            ##############################
            data_as_json = "{ \"ecg\" : "
            data_as_json = data_as_json + tostring(y_filtered)
            data_as_json = data_as_json + " \"hr\" : " +  str(heart_rate) + '}'
            # print(data_as_json)
            print('--------------------')
            send_to_server(data_as_json)

    except KeyboardInterrupt:
        print("Keyboard interupted")
        # Turn BITalino led on
        device.trigger(digitalOutput)
        # Stop acquisition
        device.stop()
        # Close connection
        device.close()

def usage(message):
    print("""

    Usage: pyhton3 collect_data [OPTIONS] -c CONFIGFILE

    -c FILENAME, --configfile FILENAME    Use FILENAME for configuration
    -h, --help                            Show help
    """)

    if(message):
        print("\nERROR: " + message + "\n\n")
    sys.exit(2)


def main():
    global WEB_HOST_ADDRESS
    global WEB_PORT
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hc:d", ["help", "configfile="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err) # will print something like "option -a not recognized"
        usage()

    configfile = None
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
        elif o in ("-c", "--configfile"):
            configfile = a
        else:
            assert False, "unhandled option"

    if(configfile is None):
        usage("Missing configfile")
    if(not os.path.exists(configfile)):
        usage("Cannot open file " + configfile)

    # read the config file.
    print("Using config file : " + configfile)
    config = configparser.ConfigParser()
    config.read(configfile)

    WEB_HOST_ADDRESS = config.get("Server", "Listen")
    WEB_PORT = config.get("Server", "Port")

    WEB_HOST_ADDRESS = str(WEB_HOST_ADDRESS)
    print(WEB_HOST_ADDRESS, WEB_PORT)


    print(macAddress,running_time,batteryThreshold,acqChannels,samplingRate,nSamples)
    print("data collection process strated")
    bitalino_data_collection()



if __name__ == "__main__":

    main()