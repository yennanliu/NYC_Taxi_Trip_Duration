# test module 
import pytest, unittest
import os 

def test_training_data_exist():
    print ('please download the train data via \n https://www.kaggle.com/c/nyc-taxi-trip-duration/data')
    print ('and put the data as : data/train.csv')
    pass 

def test_validate_data_exist():
    file_exist = os.path.isfile('data/test.csv')
    assert file_exist == True   

if __name__ == '__main__':
    pytest.main([__file__])