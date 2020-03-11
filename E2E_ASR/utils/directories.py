import os
import sys

# check and make directory
# argument should come in format of list
def CheckDir(dir_list):
    if type(dir_list)!=list:
        dir_list = [dir_list]

    for dir in dir_list:
        if not os.path.exists(dir):
            os.makedirs(dir)
    return

# check if file exists in directory
def FileExists(dir):
    if os.path.isfile(dir):
        return True
    else:
        return False

# return model directory
def ModelDir(model_save_dir, epoch, learning_rate, mtl_alpha):
    model_dir = model_save_dir + '/model_' + str(learning_rate) + 'lr_' + str(epoch) + 'ep_' + str(mtl_alpha) + 'mtl'
    return model_dir