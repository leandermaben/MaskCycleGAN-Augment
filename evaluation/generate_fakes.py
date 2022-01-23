from args.cycle_GAN_eval_arg_parser import CycleGANEvalArgParser

def main(args):

    source_id = args.speaker_A_id if args.model_name == 'generator_A2B' else args.speaker_B_id
    target_id = args.speaker_A_id if args.model_name == 'generator_B2A' else args.speaker_B_id

    for file in os.listdir(os.path.join(args.data_dir,source_id)):
        if file[-4:]!='.wav':
            print(f'Invalid Format. Skipping {file}')
            continue
        os.makedir(args.temporary_cache)


    os.system()
   

if __name__ == "__main__":
    parser = CycleGANEvalArgParser()
    args = parser.parse_args()
    main(args)