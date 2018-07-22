#!/bin/sh


function lauch_env () {

source activate ds_dash  &&  export PYTHONPATH=/Users/yennanliu/NYC_Taxi_Trip_Duration/
}


function help_ () {
	echo
		"""
################################################################
# NYC Taxi ML help script                                      #
################################################################
lauch env : 
chmod +x start.sh
source start.sh -l

check commands :
chmod +x start.sh
./start.sh -h

install library :
chmod +x start.sh
./start.sh -i 

 """
}


function install_ () {
echo "install library via pip"
pip install pandas numpy sklearn 

# https://stackoverflow.com/questions/43327020/xgboostlibrarynotfound-cannot-find-xgboost-library-in-the-candidate-path-did-y
echo "install xgboost"
cd && git clone --recursive https://github.com/dmlc/xgboost \
cd xgboost; cp make/minimum.mk ./config.mk; make -j4

echo "install lightGBM"
cd && brew install cmake && brew install gcc --without-multilib \
&& git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM \ 
export CXX=g++-7 CC=gcc-7 \
mkdir build ; cd build \ 
cmake ..   \ 
make -j4 

}



if [ -z "$1" ] || [[ "$1" != "-l"  && "$1" != "-h"  &&  "$1" != "-i" ]]
	then 
	echo "command not found, please use './start.sh -h' for more help "
fi 


case "$1"  in
	-l) lauch_env;; 
	-h) help_ ;;
	-i) install_;;
esac 






