import os
import sys

# check and make directory
# argument should come in format of list
def CheckDir(dir_list):
    if type(dir_list)!=list:
        print('directories should be aranged in type of list')
        sys.exit(1)

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
def ModelDir(model_save_dir, epoch, learning_rate):
    enc_dir = model_save_dir + '/encoder_' + str(learning_rate) + '_' + str(epoch)
    dec_dir = model_save_dir + '/decoder_' + str(learning_rate) + '_' + str(epoch)
    return enc_dir, dec_dir
