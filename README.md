Feature extraction tool for data from BITalino
==============================================

\textit{BITalino board} \footnote{http://bitalino.com/en/} is a noninvasive biomedical signal acquisition device. This device comprises of micro-controller unit, Bluetooth (BT) module for wireless communication and power block with a Lithium Ion Polymer battery\cite{ch401} (Figure \ref{fig:bitalino_board}). This device offers measurement sensors for bioelectrical and biomechanical data acquisition. BITalino board along with the respective sensors enables acquisition of electrocardiogram (ECG), electromyogram (EMG), electrodermal activity (EDA) and electroencephalogram (EEG) 

version number: 0.1.0

compatible with > python 3.5

Dependancies
------------

    $ pip3 install numpy
    $ pip3 install biosppy
    $ pip3 install matplotlib
    
    
Usage
-----
    $pyhton3 collect_data [OPTIONS] -c CONFIGFILE
    
    -c FILENAME, --configfile FILENAME    Use FILENAME for configuration
    -h, --help                            Show help 

