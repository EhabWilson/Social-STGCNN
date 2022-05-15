import sys
import os
import datetime

len = 32
tag = ""
sys.path.append(os.getcwd())
current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
import socket
test_dir = os.path.join('./runs', current_time + '_' + socket.gethostname() + "_" +str(len))
log_dir = os.path.join(
    './runs', current_time + '_' + socket.gethostname() + "dict_len_" + str(len) + '/' +tag)
print(test_dir)
print(log_dir)