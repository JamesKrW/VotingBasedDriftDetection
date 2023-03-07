from utils import loadjson
import pickle

if __name__=='__main__':
    src_path='/home/cc/github/ref-sum/refsum/data/tmptrain_v1.json'
    pickle_path='/home/cc/github/ref-sum/refsum/data/tmptrain_cite.pickle'
    data=loadjson(src_path)
    citepair=[]
    for paper in data.keys():
        for arxiv_id in data[paper]['cite']:
            citepair.append((paper,arxiv_id))
    with open(pickle_path,'wb') as f:
        pickle.dump(citepair,f)
    print(len(citepair))

    csv_path='/home/cc/github/ref-sum/refsum/data/tmptrain_cite.csv'
    with open(csv_path,'w') as f:
        for item in citepair:
            f.write(f"{item[0]},{item[1]}\n")

    
    with open(pickle_path,'rb') as f:
        test_pair=pickle.load(f)
    for item in test_pair:
        assert item[0] in data.keys() and item[1] in data.keys()

    