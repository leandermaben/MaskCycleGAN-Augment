import argparse
import shutil
import torch
import tqdm
import os
import numpy as np
import math
import cv2
import librosa
import pickle
import matplotlib.pyplot as plt
from mask_cyclegan_vc.utils import save_pickle

"""
Code to transfer audio data from a source folder to a target folder with train and test splits.
Change AUDIO_DATA_PATH_DEFAULT to point to root dir such that (or use command line argument)
-root
    -subdirectory[0]
        -sample_0
        -....
    -subdirectory[1]
        -sample_0
        -sample_1
Change CACHE_DEFAULT to the directory where you want data to be stored.

There are 2 options to transfer data -> It can be transferred as audio files or as spectrograms.
Spectrograms are generated using MelGAN.However, MelGAN performs poorly when converting noisy spectrogram back to audio.
Hence it is recommended to use the 'audio' option for the argument --transfer_mode (It is already set as default)

"""

AUDIO_DATA_PATH_DEFAULT = '/content/drive/MyDrive/NTU - Speech Augmentation/Parallel_speech_data'
SUBDIRECTORIES_DEFAULT = ['clean','noisy']
CACHE_DEFAULT = '/content/MaskCycleGAN-Augment/data_cache'


def get_filenames(fileNameA):
    """
    Custom function for this specific dataset.
    It returns the names of corresponding files in the 2 classes along with the common name by which it should be saved.

    Args:
    fileNameA(str) : Filename in the first class

    Created By Leander Maben
    """

    return fileNameA, fileNameA[:32]+'-A.wav', fileNameA[:32]+'.wav'



def transfer_aligned_audio_raw(root_dir,class_ids,data_cache,train_percent,test_percent):
    """
    Transfer audio files to a convinient location for processing with train,test,validation split.

    Important Note: The splitting of data by percent is based on file numbers and not on cummulative duration
    of clips. Moreover, it does not take into the account the number of clips that are discarded for being less than 1 second long.

    Arguments:
    root_dir(str) - Root directory where files of specified classes are present in subdirectories.
    class_id(str) - Current class ID of data objects
    data_cache(str) - Root directory to store data
    train_percent(int) - Percent of data clips in train split
    test_percent(int) - Percent of data clips in test split

    Created By Leander Maben. 
    """

    for class_id in class_ids:
        os.makedirs(os.path.join(data_cache,class_id,'train'))
        os.makedirs(os.path.join(data_cache,class_id,'test'))

    files_list = [x for x in os.listdir(os.path.join(root_dir,class_ids[0])) if x[-4:]=='.wav']
    num_files = len(files_list)

    indices = np.arange(0,num_files)
    np.random.seed(7)
    np.random.shuffle(indices)

    train_split = math.floor(train_percent/100*num_files)
    test_split = math.floor(test_percent/100*num_files)

    for phase,(start, end) in zip(['train','test'],[(0,train_split),(num_files-test_split,num_files)]):
        duration = 0
        clips = 0
        for i in range(start,end):
            fileA, fileB, file=get_filenames(files_list[indices[i]])
            if librosa.get_duration(filename=os.path.join(root_dir,class_ids[0],fileA)) < 1: #Skipping very short files
                continue
            shutil.copyfile(os.path.join(root_dir,class_ids[0],fileA),os.path.join(data_cache,class_ids[0],phase,file))
            shutil.copyfile(os.path.join(root_dir,class_ids[1],fileB),os.path.join(data_cache,class_ids[1],phase,file))
            duration+=librosa.get_duration(filename=os.path.join(data_cache,class_ids[0],phase,file))
            clips+=1
        print(f'{duration} seconds ({clips} clips) of Audio saved to {phase}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate Spectrograms')
    parser.add_argument('--audio_data_path', dest = 'audio_path', type=str, default=AUDIO_DATA_PATH_DEFAULT, help="Path to audio root folder")
    parser.add_argument('--source_sub_directories', dest = 'sub_directories',type=str, default=SUBDIRECTORIES_DEFAULT, help="Sub directories for data")
    parser.add_argument('--data_cache', dest='data_cache', type=str, default=CACHE_DEFAULT, help="Directory to Store data and meta data.")
    parser.add_argument('--train_percent', dest='train_percent', type=int, default=70, help="Percentage for train split")
    parser.add_argument('--test_percent', dest='test_percent', type=int, default=15, help="Percentage for test split")
    
    args = parser.parse_args()

    for arg in vars(args):
        print('[%s] = ' % arg, getattr(args, arg))

    transfer_aligned_audio_raw(args.audio_path,args.sub_directories,args.data_cache,args.train_percent,args.test_percent)




