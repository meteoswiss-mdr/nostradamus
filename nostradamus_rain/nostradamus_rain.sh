#!/bin/bash

# go to operational pytroll folder  
#echo ''
. /opt/users/$LOGNAME/monti-pytroll/setup/bashrc no_virtual_environment
#export python=/usr/bin/python
#export python=/opt/users/common/packages/anaconda3/envs/PyTroll_$LOGNAME/bin/python

cd $PYTROLLHOME/packages/nostradamus

echo "*** Start to make seviri pictures (loop until all data is there)"
echo 
python nostradamus_rain.py input_rr.py
