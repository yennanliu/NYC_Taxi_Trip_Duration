# -*- coding: utf-8 -*-

# script for model I/O 
import pickle 
from datetime import datetime


today_ = datetime.now().strftime("%m%d")
print (today_)

# save model 
def save_model(model):
    try:
        #with open('../model/model_{}.pkl'.format(today_), 'wb') as fid:
        with open('model/model_{}.pkl'.format(today_), 'wb') as fid:
            pickle.dump(model, fid)
            print ('model save success')
    except:
        print ('saving fail')



# load model 
def load_model():
	try:       
		#with open('../model/model_0730.pkl', 'rb') as fid:  
		with open('model/model_{}.pkl'.format(today_), 'rb') as fid:      
			loaded_model = pickle.load(fid)     
			return loaded_model
	except:
		print ('model load fail')


