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
    dd=DriftDetection(ogdata)
    dd.add_method(ADWIN(),1)
    dd.add_method(DDM(),1)
    dd.add_method(EDDM(),1)
    dd.add_method(HDDM_A(),1)
    dd.add_method(HDDM_W(),1)
    dd.add_method(PageHinkley(),1)
    dd.add_method(KSWIN(),1)
    window_size=75
    thresh_hold=300
    change_list=dd.get_voted_drift(window_size=window_size,thresh_hold=thresh_hold)
    change_list=[i+1000 for i in change_list]
    rec_time=int(time.time())
    plot(ogdata,change_list,f'{args.data_path}/og_{rec_time}_{window_size}_{thresh_hold}.png')

if __name__=='__main__':
    og_path_list=["m_1.csv","m_14.csv","m_32.csv","m_42.csv","m_55.csv","m_98.csv","m_155.csv","m_259.csv","m_350.csv","m_450.csv"]

    data_path_list=["m_1/2023-03-08T03-59-13","m_14/2023-03-08T03-59-24","m_32/2023-03-08T03-59-31",
                    "m_42/2023-03-08T03-59-36","m_55/2023-03-08T03-59-42","m_98/2023-03-08T03-59-47",
                    "m_155/2023-03-08T03-59-52","m_259/2023-03-08T04-00-03","m_350/2023-03-08T04-00-10","m_450/2023-03-08T04-00-16"]
    i=9
    parser = argparse.ArgumentParser(description='training template')
    parser.add_argument('--og_path', type=str, default=f'/home/cc/github/VotingBasedDriftDetection/methods/data/{og_path_list[i]}', metavar='N',)
    parser.add_argument('--data_path', type=str, default=f'/home/cc/github/VotingBasedDriftDetection/{data_path_list[i]}', metavar='N',)
    args = parser.parse_args()
    main(args)