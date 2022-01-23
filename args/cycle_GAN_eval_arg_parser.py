from args.cycleGAN_test_arg_parser import CycleGANTestArgParser 

DATA_DIRECTORY_DEFAULT = '/content/drive/MyDrive/NTU - Speech Augmentation/Parallel_speech_data'
SPEAKER_IDS_DEFAULT = ['clean', 'noisy']

class CycleGANEvalArgParser(CycleGANTestArgParser):
    def __init__(self):
        super(CycleGANEvalArgParser,self).__init__()
        self.parser.add_argument('--data_directory', type='str', default=DATA_DIRECTORY_DEFAULT, help = 'Data directory for audio clips')
        