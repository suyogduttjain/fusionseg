import os
import sys

data_dir = './sample/'

#Run the appearance stream
cmd = 'python2.7 fusionseg.py appearance jpg ' + data_dir
print(cmd)
os.system(cmd)

#Run the motion stream
cmd = 'python2.7 fusionseg.py motion png ' + data_dir
print(cmd)
os.system(cmd)

#Run the fusionseg joint model trained on DAVIS train subset
cmd = 'python2.7 fusionseg_joint.py joint_davis_train ' + data_dir
print(cmd)
os.system(cmd)

#Run the fusionseg joint model trained on DAVIS val subset
cmd = 'python2.7 fusionseg_joint.py joint_davis_val ' + data_dir
print(cmd)
os.system(cmd)
