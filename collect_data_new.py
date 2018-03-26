from local_bitalino import BITalino
import time,datetime
import numpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import biosppy as bs # maybe not used!

import scipy.io as sio
from scipy import signal
import scipy

from socket import socket
import peakutils
import configparser
import getopt
import sys,os
import math

delete_me = []

WEB_HOST_ADDRESS = ""
WEB_PORT = 1234
plot = True
send_flag = False
save_raw_data = False

lowcut = 30
highcut = 200

labels = ["'nSeq'", "'I1'", "'I2'", "'O1'", "'O2'", "'A1'", "'A2'", "'A3'", "'A4'", "'A5'", "'A6'"]

# initial settings - default settings
macAddress = '20:16:12:22:01:28'
running_time = 30
batteryThreshold = 30
acqChannels = [0,1] # 1 for A2 | 0 - A1
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

    #connect to the socket
    sock.connect((WEB_HOST_ADDRESS,int(WEB_PORT)))

    # send the data as a json
    sock.send(data_as_json.encode('utf-8'))

    # close the connection.
    sock.close()

def butter_bandpass(lowcut, highcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_highpass(cutoff, fs, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def getpeak(ecg, time):
    '''detects the peak from the array!'''
    if(len(ecg) < 500):
        return [0]
    else:
        indexes = peakutils.indexes(ecg, thres=0.8 , min_dist=2)
        return indexes


def get_time_domain_features(ecg,peaklist,fs):
    '''

    :param ecg: our filtered signal
    :param peaklist: array of indexes which gives th peak
    :param fs: sampling frequency
    :return: ??
    '''
    RR_list = []
    cnt = 0
    while (cnt < (len(peaklist) - 1)):
        RR_interval = (peaklist[cnt + 1] - peaklist[cnt])  # Calculate distance between beats in # of samples
        s_dist = (RR_interval / fs)
        RR_list.append(s_dist)  # Append to list
        cnt += 1

    hr = 60 / np.mean(RR_list)   # 60sec (1 minute) / average R-R interval of signal * (new sample arrives).


    return hr


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

def ecg_bin_to_millivolts(ecg):
    '''
        Vcc =  battery voltage = 3.7 V | Sensor_gain = 1100
        RMOhm = 1 - EDAB / 2^n (sensor resistance in mega ohms)
        EDAS = 1 / RMOhm (conductance in microsiemens)
        Reference : http://forum.bitalino.com/viewtopic.php?f=12&t=128

        :param eda: eda array
        :return:
        '''

    ecg_value_millivolts = []
    for i in range(0, len(ecg)):
        x = ecg[i]/1024 - (0.5) * 3.3
        x = x/1100
        x = x * 1000
        ecg_value_millivolts.append(x)

    return ecg_value_millivolts


def eda_process(eda):
    pass


def write_to_file(filename,raw_data):
    with open(filename, 'ab') as f:
        for line in raw_data:
            a = numpy.array(line)
            np.savetxt(f, a.reshape(1, a.shape[0]) , delimiter=', ' ,fmt="%5f")

def bitalino_data_collection():
    '''
    The core function of the file.
    :return:
    '''

    Fs = float(int(samplingRate)/10)

    szplot = 500  # to show the plot (show for last ) # our window size!
    instantaneous_sample = 1000 # this represents the time to get the peak data.(1000saples  = 10 seconds)
    heart_rate = 40

    # Connect to BITalino
    device = BITalino(macAddress)
    print("device connected to bitalino")

    # Set battery threshold
    device.battery(batteryThreshold)

    print("SAmpling Rate" + str(samplingRate))
    # Start Acquisition
    device.start(samplingRate, acqChannels)


    # time initialization
    timeend = 0.0
    timeinit = 0.0  # initial time
    timeend += float(nSamples) / float(Fs)  # end time ( in our case its 100/1000 = 0.1 sec)


    time_elapsed = []

    ecg = []
    eda = []
    peakind = [] # for peak detection

    ecg_data = []
    eda_data = []


    # plotting
    if(plot) :
        fig = plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111)

        plt.ion()
        plt.xlabel('Time (seconds)')
        line0, = ax.plot(time_elapsed, ecg_data, 'y-', label='RAW data')  # raw data
        line1, = ax.plot(time_elapsed, ecg, 'b-' , alpha=0.3, label='detrended RAW data')  # raw data
        line2, = ax.plot(time_elapsed, ecg, 'g-', alpha=0.7 ,label='filtered data')  # to represent teh filtered data
        line3, = ax.plot(time_elapsed, ecg, 'ro' , label='detected peak') # peaks
        fig.show()
        fig.canvas.draw()


        fig1 = plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        ax_eda = fig1.add_subplot(111)
        plt.ion()
        plt.xlabel('Time (seconds)')
        plt.ylabel('Conductance (microSiemens)')
        line,  = ax_eda.plot(time_elapsed, eda, 'r-' , label='eda RAW') # peaks
        lineeda1, = ax_eda.plot(time_elapsed, eda, 'b-', label=' eda filtered')  # peaks
        fig1.show()
        fig1.canvas.draw()


    filename = 'raw_data/recording_'+ str(datetime.datetime.now()) + '.txt'

    time_in_start = time.time()
    try:
        # indefinite signal capture
        while 1:



            # read data from the device
            received_data = device.read(nSamples)
            # print("received_data_size : " + str(len(received_data)))


            passed_time = time.time() - time_in_start
            print("real time :" + str(passed_time) + " sec")

            # update time
            time_elapsed = np.concatenate((time_elapsed, np.linspace(timeinit, timeend, nSamples + 1)[1:]), 0)
            asda = len(time_elapsed) / Fs
            print("time from samples : " + str(asda) + " sec")

            timeinit = time_elapsed[-1]
            # print("start time : " + str(timeend))
            timeend += float(nSamples) / float(Fs)

            print("_________________________________________________")

            if(save_raw_data):
                write_to_file(filename,received_data)

            ecg_data = np.concatenate((ecg_data, ecg_bin_to_millivolts(received_data[:, -1])), axis=0)
            eda_data = np.concatenate((eda_data, eda_bin_to_microsiemens(received_data[:, -2])), axis=0)


            # we detrend the data for heart rate
            ecg = signal.detrend(ecg_data)

            # we convert the data from binary to micro siemens.
            eda_raw = eda_data

            # highpassfilter for EDA
            ale,ble = butter_highpass(0.05, samplingRate) # high pass cutoff = 0.05 Hz
            eda = signal.filtfilt(ale, ble, eda_raw);


            #bandpassfilter for ECG
            ale,ble = butter_bandpass(lowcut, highcut , samplingRate)
            ecg_filtered = signal.filtfilt(ale, ble, ecg);






            # update plot everytime you recive the data
            # note that we show the user past 500 data samples and hence data from past 0.5 second = 500 msec(millisec)
            x = time_elapsed[-szplot:]
            y_raw = ecg[-szplot:]
            y_filtered = ecg_filtered[-szplot:]

            peakind = getpeak(y_filtered,x)

            # #  some adjustments to plot the data
            x_peaks = [x[i] for i in peakind]  # peak time
            y_peaks = [y_filtered[i] for i in peakind]  # peak value

            heart_rate = get_time_domain_features(y_filtered,peakind,Fs)

            print(heart_rate)

            if math.isnan(heart_rate):
                heart_rate= 40

            #detected_value = bs.ecg.ecg(signal=y_filtered, sampling_rate=100.0, show=False)
            #print(detected_value['heart_rate'])

            if (len(time_elapsed)/Fs > 10):
                ecg_data_for_peak_detection = ecg_filtered[-instantaneous_sample:]
                time_for_peak_detection = time_elapsed[-instantaneous_sample:]


                # we now find peaks for past 10 seconds ( R peak detection)
                peakind_new = getpeak(ecg_data_for_peak_detection, time_for_peak_detection)

                heart_rate_new = get_time_domain_features(ecg_data_for_peak_detection, peakind_new, Fs)

                print("new HR  :" + str(heart_rate_new))







            if (plot):
                line0.set_data(x, ecg_data[-szplot:])
                line1.set_data(x,y_raw)
                line2.set_data(x, y_filtered)
                line3.set_data(x_peaks,y_peaks)
                # line4.set_data(x_peaks_hamilton, y_peaks_hamilton)

                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                ax.legend(loc='upper right',handles=[line0,line1,line2,line3]) # to add the legend.
                plt.draw()

                line.set_data(x, eda_raw[-szplot:])
                lineeda1.set_data(x,eda[-szplot:])
                ax_eda.relim()
                ax_eda.autoscale_view()
                fig1.canvas.draw()
                ax_eda.legend(loc='upper right',handles=[line,lineeda1])  # to add the legend.
                plt.draw()

            # send data to server as a json
            # note we  send the last 500
            #{ "ecg" : "[data]" ,
            #   "fatures" = [hr,other?],
            #  "eda" = "[eda data]"
            # }
            ##############################
            data_as_json = "{ \"ecg\" : "
            data_as_json = data_as_json + tostring(y_filtered) + ','
            data_as_json = data_as_json + " \"ecg_features\" : " +  str(heart_rate) + '}'



            # we initially send ecg data
            if send_flag:
                send_to_server(data_as_json)
            print('data sent to the web server...')

            # prep eda data
            eda_data_as_json = "{ \"eda\" : "
            eda_data_as_json = eda_data_as_json + tostring(eda[-szplot:]) + '}'

            if send_flag:
                send_to_server(eda_data_as_json)
                print('data sent to the web server...')



    except KeyboardInterrupt:
        print("Keyboard interupted")

        x =[]
        for i in range(1,len(delete_me)):
            x.append(i)

        plt.plot(x,delete_me)
        plt.show()

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