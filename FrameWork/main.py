import numpy as np
from driftdetection import DriftDetection
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection import DDM
from skmultiflow.drift_detection.eddm import EDDM
from skmultiflow.drift_detection.hddm_a import HDDM_A
from skmultiflow.drift_detection.hddm_w import HDDM_W
from skmultiflow.drift_detection import PageHinkley
from skmultiflow.drift_detection import KSWIN
import pickle
from utils import plot
import time
def main():
    ogdata = np.genfromtxt('/home/cc/github/VotingBasedDriftDetection/data/avg_all.csv', delimiter=',')

  
    ogdata=ogdata[1:,-2]

    with open('/home/cc/github/VotingBasedDriftDetection/methods/results/MLP-1678242665.pkl', 'rb') as f:
    # Use pickle to deserialize the NumPy array from the file
        data_stream = pickle.load(f)
    
    # data_stream=data[1:,-2]
    dd=DriftDetection(data_stream)
    dd.add_method(ADWIN(),1)
    dd.add_method(DDM(),1)
    dd.add_method(EDDM(),1)
    dd.add_method(HDDM_A(),1)
    dd.add_method(HDDM_W(),1)
    dd.add_method(PageHinkley(),2)
    dd.add_method(KSWIN(),1)
    window_size=50
    thresh_hold=200
    change_list=dd.get_voted_drift(window_size=window_size,thresh_hold=thresh_hold)
    rec_time=int(time.time())
    plot(ogdata,change_list,f'/home/cc/github/VotingBasedDriftDetection/FrameWork/results/mlp_{rec_time}_{window_size}_{thresh_hold}.png')

if __name__=='__main__':
    main()