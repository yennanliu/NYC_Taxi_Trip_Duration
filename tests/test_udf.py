# test module 
import pytest, unittest
# import to-test udf (user defined function) 
import sys
sys.path.append(".")
from run.train import get_haversine_distance, get_manhattan_distance,\
                      get_direction

def test_get_haversine_distance():
    lat1, lng1, lat2, lng2 = -73.98,40.73,-73.99,40.75 
    h = get_haversine_distance(lat1, lng1, lat2, lng2)
    assert h ==  1.2699896429084849

def test_get_manhattan_distance():
    lat1, lng1, lat2, lng2 = -73.98,40.73,-73.99,40.75 
    a = get_haversine_distance(lat1, lng1, lat1, lng2)
    b = get_haversine_distance(lat1, lng1, lat2, lng1) 
    assert a+b == 1.7256849524059557

def test_get_direction():
    lat1, lng1, lat2, lng2 = -73.98,40.73,-73.99,40.75 
    direction = get_direction(lat1, lng1, lat2, lng2)
    assert direction == 151.1206636530826

if __name__ == '__main__':
    pytest.main([__file__])