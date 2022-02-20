import os
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.utils.data as data
import torchaudio

from mask_cyclegan_vc.model import Generator, Discriminator
from args.cycleGAN_test_arg_parser import CycleGANTestArgParser
from dataset.vc_dataset import VCDataset
from mask_cyclegan_vc.utils import decode_melspectrogram
from logger.train_logger import TrainLogger
from saver.model_saver import ModelSaver

from mask_cyclegan_vc.utils import denorm_and_numpy, getTimeSeries
import soundfile as sf


class MaskCycleGANVCTesting(object):
    """Tester for MaskCycleGAN-VC
    """

    def __init__(self, args):
        """
        Args:
            args (Namespace): Program arguments from argparser
        """
        # Store Args
        self.device = args.device

        args.num_threads = 0   # test code only supports num_threads = 0
        args.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        args.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    
        
        if hasattr(args,'eval') and args.eval:
            self.eval=True
            os.makedirs(args.eval_save_dir, exist_ok=True)
            self.eval_save_path = os.path.join(args.eval_save_dir,args.filename)
        else:
            self.eval=False
            self.converted_audio_dir = os.path.join(args.save_dir, args.name, 'converted_audio')
            os.makedirs(self.converted_audio_dir, exist_ok=True)

        self.model_name = args.model_name

        self.speaker_A_id = args.speaker_A_id
        self.speaker_B_id = args.speaker_B_id


        self.sample_rate = args.sample_rate

        self.dataset = NoiseDataset(args)

        self.test_dataloader = torch.utils.data.DataLoader(dataset=self.dataset,
                                                           batch_size=1,
                                                           shuffle=False,
                                                           drop_last=False)

        # Generator
        self.generator = Generator().to(self.device)
        self.generator.eval()

        # Load Generator from ckpt
        self.saver = ModelSaver(args)
        self.saver.load_model(self.generator, self.model_name)
        
    def save_audio(opt, visuals_list, img_path):

        """
        Borrowed from https://github.com/shashankshirol/GeneratingNoisySpeechData
        """

        results_dir = os.path.join(opt.save_dir, opt.name)
        img_dir = os.path.join(results_dir, 'audios')
        short_path = ntpath.basename(img_path[0])
        name = os.path.splitext(short_path)[0]

        label = "fake_B"  # Concerned with only the fake generated; ignoring other labels

        file_name = '%s/%s.wav' % (label, name)
        os.makedirs(os.path.join(img_dir, label), exist_ok=True)
        save_path = os.path.join(img_dir, file_name)

        flag_first = True

        for visual in visuals_list:
            im_data = visual #Obtaining the generated Output
            im = denorm_and_numpy(im_data) #De-Normalizing the output tensor to reconstruct the spectrogram

            #Resizing the output to 129x128 size (original splits)
            if(im.shape[-1] == 1): #to drop last channel
                im = im[:,:,0]
            im = Image.fromarray(im)
            im = im.resize((128, 129), Image.LANCZOS)
            im = np.asarray(im).astype(np.float)

            if(flag_first):
                spec = im
                flag_first = False
            else:
                spec = np.concatenate((spec, im), axis=1) #concatenating specs to obtain original.

        data, sr = getTimeSeries(spec, img_path, opt.spec_power, opt.energy, state = opt.phase)
        sf.write(save_path, data, sr)

        return

    def loadPickleFile(self, fileName):
        """Loads a Pickle file.

        Args:
            fileName (str): pickle file path

        Returns:
            file object: The loaded pickle file object
        """
        with open(fileName, 'rb') as f:
            return pickle.load(f)

    def test(self):

        ds_len = len(self.dataset)
        idx = 0
        datas = []

        for i, data in enumerate(self.test_dataloader):
            datas.append(data)
        while idx < ds_len:

            if(idx >= opt.num_test):
                break

            if self.model_name == 'generator_A2B':
                real = datas[idx]['A']
                mask = datas[idx]['A_mask']
                img_path = datas[idx]['A_paths']
            else:
                real = datas[idx]['B']
                mask = datas[idx]['B_mask']
                img_path = datas[idx]['B_paths']
            fake = self.generator(real, mask)
            visuals_list = [fake]
            num_comps = datas[idx]["A_comps"] ##Need to generalize for bidirectional
            comps_processed = 1

            while(comps_processed < num_comps):
                idx += 1
                if self.model_name == 'generator_A2B':
                    real = datas[idx]['A']
                    mask = datas[idx]['A_mask']
                    img_path = datas[idx]['A_paths']
                else:
                    real = datas[idx]['B']
                    mask = datas[idx]['B_mask']
                    img_path = datas[idx]['B_paths']
                fake = self.generator(real, mask)
                visuals_list.append(fake)
                comps_processed += 1

            print("saving: ", img_path[0])
            save_audio(opt, visuals_list, img_path)
            idx += 1



if __name__ == "__main__":
    parser = CycleGANTestArgParser()
    args = parser.parse_args()
    tester = MaskCycleGANVCTesting(args)
    tester.test()
