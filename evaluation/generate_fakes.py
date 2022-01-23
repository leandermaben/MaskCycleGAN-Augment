from args.cycle_GAN_eval_arg_parser import CycleGANEvalArgParser
import os
import shutil
from data_preprocessing.preprocess_vcc2018 import preprocess_dataset
from mask_cyclegan_vc.test import MaskCycleGANVCTesting
import librosa

def main(args):

    source_id = args.speaker_A_id if args.model_name == 'generator_A2B' else args.speaker_B_id
    target_id = args.speaker_A_id if args.model_name == 'generator_B2A' else args.speaker_B_id

    # Creating temporary cache for data

        source_orig_data_path = os.path.join(args.eval_cache,'orig',source_id)
        target_orig_data_path = os.path.join(args.eval_cache,'orig',target_id)
        source_processed_data_path = os.path.join(args.eval_cache,'single_processed',source_id)
        target_processed_data_path = os.path.join(args.eval_cache,'single_processed'target_id)
        source_agg_processed_path = os.path.join(args.eval_cache,'agg',source_id)
        target_agg_processed_path = os.path.join(args.eval_cache,'agg',target_id)

        os.makedirs(source_orig_data_path)
        os.makedirs(target_orig_data_path)
        os.makedirs(args.eval_cache,'converted_audio','real')

    #Preprocess all clips for aggregate statistics

    for speaker_id in [source_id,target_id]:
        preprocess_dataset(data_path=args.data_directory, speaker_id=speaker_id,
                           cache_folder=os.path.join(args.eval_cache,'agg'))
        shutil.remove(os.path.join(args.eval_cache,'agg',speaker_id,f'{speaker_id}_normalized.pickle'))

    for file in os.listdir(os.path.join(args.data_directory,source_id)):

        # Verifying format

        if file[-4:]!='.wav':
            print(f'Invalid Format. Skipping {file}')
            continue

        #Copy Data

        shutil.copyfile(os.path.join(args.data_directory,source_id,file),os.path.join(source_orig_data_path,file))
        shutil.copyfile(os.path.join(args.data_directory,target_id,file),os.path.join(target_orig_data_path,file))

        #Preprocess Data

        for speaker_id in [source_id,target_id]:
            preprocess_dataset(data_path=os.path.join(args.eval_cache,'orig',speaker_id),
                                    speaker_id=speaker_id,cache_folder=os.path.join(args.eval_cache,'processed'))
            shutil.remove(source_orig_data_path = os.path.join(args.eval_cache,'processed',speaker_id),
                                                    f'{speaker_id}_norm_stat.npz') #Removing individual stats
            shutil.copyfile(os.path.join(os.path.join(args.eval_cache,'agg',speaker_id,f'{speaker_id}_norm_stat.npz'),
                                    os.path.join(args.eval_cache,'processed',speaker_id,f'{speaker_id}_norm_stat.npz')) #Copying aggregated stats
        
        # Run inference
        args.eval = True
        args.eval_save_path = os.path.join(args.eval_cache,'converted_audio','generated')
        args.filename = file
        args.preprocessed_data_dir = os.path.join(args.eval_cache,'processed')
        tester = MaskCycleGANVCTesting(args)
        tester.test()

        #Copy original target file to Real folder with given sample_rate
        real , sr = librosa.load(os.path.join(target_orig_data_path,file))
        librosa.output.write_wav(os.path.join(args.eval_cache,'converted_audio','real',file),real,22050)
        shutil.copyfile(os.path.join(os.path.join(args.eval_cache,'orig',target_id,f'{speaker_id}_norm_stat.npz'),
                                    os.path.join(args.eval_cache,'converted_audio',speaker_id,f'{speaker_id}_norm_stat.npz'))
        
        #Deleting Processed and Orig Directories
        shutil.rmtree(source_orig_data_path)
        shutil.rmtree(target_orig_data_path)
        shutil.rmtree(source_processed_data_path)
        shutil.rmtree(target_processed_data_path)
        

if __name__ == "__main__":
    parser = CycleGANEvalArgParser()
    args = parser.parse_args()
    main(args)