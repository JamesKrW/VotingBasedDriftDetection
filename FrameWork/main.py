import numpy as np
from driftdetection import DriftDetection
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection import DDM
from skmultiflow.drift_detection.eddm import EDDM
from skmultiflow.drift_detection.hddm_a import HDDM_A
from skmultiflow.drift_detection.hddm_w import HDDM_W
from skmultiflow.drift_detection import PageHinkley
from skmultiflow.drift_detection import KSWIN
from utils import plot
def main():
    data = np.genfromtxt('./data/avg_all.csv', delimiter=',')
    data_stream=data[1:,-2]
    dd=DriftDetection(data_stream)
    dd.add_method(ADWIN(),1)
    dd.add_method(DDM(),1)
    dd.add_method(EDDM(),1)
    dd.add_method(HDDM_A(),1)
    dd.add_method(HDDM_W(),1)
    dd.add_method(PageHinkley(),2)
    dd.add_method(KSWIN(),1)
    window_size=25
    thresh_hold=100
    change_list=dd.get_voted_drift(window_size=window_size,thresh_hold=thresh_hold)
    plot(data_stream,change_list,f'./test_{window_size}_{thresh_hold}.png')

if __name__=='__main__':
    main()