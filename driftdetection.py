import numpy as np
from collections import deque


class DriftDetection:
    def __init__(self,data_stream):
        self.methods=[]
        self.weights=[]
        self.drifts=[]
        self.data_stream=data_stream
    
    def add_method(self,method,weight):
        self.methods.append(method)
        self.weights.append(weight)
        self.drifts.append(deque())


    def get_drift_point(self):
        for pos,ele in enumerate(self.data_stream):
            for method_index,method in enumerate(self.methods):
                method.add_element(ele)
                if method.detected_change():
                    self.drifts[method_index].append(pos)
        # for idx,drift in enumerate(self.drifts):
        #     print(len(drift))
    
    def vote_drift(self,window_size,thresh_hold):
        self.vote_drift=[]
        for i in range(0,len(self.data_stream),window_size):
            pos_sum=0
            weight_sum=0
            for method_index,method in enumerate(self.methods):
                while len(self.drifts[method_index])!=0: 
                    pos=self.drifts[method_index][0]
                    if pos>=(i+1)*window_size:
                        break
                    else:
                        pos_sum+=self.weights[method_index]*pos
                        weight_sum+=self.weights[method_index]
                        self.drifts[method_index].popleft()
            if weight_sum!=0:
                print(weight_sum)
            if weight_sum>thresh_hold:
                mean_pos=int(pos_sum/weight_sum)
                self.vote_drift.append(mean_pos)

    def get_voted_drift(self,window_size,thresh_hold):
        for method_idx,item in enumerate(self.drifts):
            self.drifts[method_idx]=deque()
        self.get_drift_point()
        self.vote_drift(window_size,thresh_hold)
        return self.vote_drift





