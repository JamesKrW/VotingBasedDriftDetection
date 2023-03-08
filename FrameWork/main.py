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
import argparse
import time
def main(args):
    ogdata = np.genfromtxt(args.og_path, delimiter=',')

  
    ogdata=ogdata[1:,-2]

    with open(f'{args.data_path}/pred.pickle', 'rb') as f:
    # Use pickle to deserialize the NumPy array from the file
        data_stream = pickle.load(f)
    
    # data_stream=data[1:,-2]
    dd=DriftDetection(data_stream)
    dd.add_method(ADWIN(),1)
    dd.add_method(DDM(),1)
    dd.add_method(EDDM(),1)
    dd.add_method(HDDM_A(),1)
    dd.add_method(HDDM_W(),1)
    dd.add_method(PageHinkley(),1)
    dd.add_method(KSWIN(),1)
    window_size=50
    thresh_hold=10
    change_list=dd.get_voted_drift(window_size=window_size,thresh_hold=thresh_hold)
    rec_time=int(time.time())
    plot(ogdata,change_list,f'{args.data_path}/mlp_{rec_time}_{window_size}_{thresh_hold}.png')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='training template')
    parser.add_argument('--og_path', type=str, default='/home/cc/github/VotingBasedDriftDetection/methods/data/m_1.csv', metavar='N',)
    parser.add_argument('--data_path', type=str, default='/home/cc/github/VotingBasedDriftDetection/m_1/2023-03-08T03-59-13', metavar='N',)
    args = parser.parse_args()
    main(args)