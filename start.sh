#!/bin/sh


function lauch_env () {

source activate g_dash  &&  export PYTHONPATH=/Users/yennanliu/NYC_Taxi_Trip_Duration/
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

pip install pandas numpy sklearn xgboost 

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






