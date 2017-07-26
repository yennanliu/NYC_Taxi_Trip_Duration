# -*- coding: utf-8 -*-

# script for model I/O 
import pickle 



# save model 
def save_model(model):
    try:
        with open('../model/model_0726.pkl', 'wb') as fid:
            pickle.dump(model, fid)
            print ('model save success')
    except:
        print ('saving fail')



# load model 
def load_model():       
    with open('../model/model_0726.pkl', 'rb') as fid:        
        loaded_model = pickle.load(fid)     
        return loaded_model