# -*- coding: utf-8 -*-

# script for model I/O 



# save model 
def save_model(model):
    try:
        with open('~/NYC_Taxi_Trip_Duration/model/model_0726.pkl', 'wb') as fid:
            pickle.dump(model, fid)
            print ('model save success')
    except:
        print ('saving fail')


# load model 
def load_model():       
    with open('~/NYC_Taxi_Trip_Duration/model/model_0726.pkl', 'rb') as fid:        
        loaded_model = pickle.load(fid)     
        return loaded_model