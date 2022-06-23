import os
import pandas as pd
import shutil
import sys
from metrics.lsd import main as lsd
from metrics.mssl import main as mssl

def run(command):
    print(command)
    exit_status = os.system(command)
    if exit_status > 0:
        exit(1)

def log(path,info):
    """
    Created by Leander Maben.
    """
    df=pd.read_csv(path)
    df.loc[len(df.index)] = info
    df.to_csv(path,index=False)

def validate(name, epochs, data_cache, results_dir):
    info = {'avg_val_lsd':[],'std_val_lsd':[],'avg_val_mssl':[],'std_val_mssl':[]}
    min_lsd=sys.maxsize
    min_mssl=sys.maxsize
    min_lsd_epoch=-1
    min_mssl_epoch=-1

    for epoch in epochs:
        run(f'python -W ignore::UserWarning -m mask_cyclegan_vc.test --name {name} --split val --save_dir results --gpu_ids 0 --speaker_A_id clean --speaker_B_id noisy --ckpt_dir /content/drive/MyDrive/APSIPA/Results/{name}/ckpts --load_epoch {epoch} --model_name generator_A2B')
        avg_lsd,std_lsd = lsd(os.path.join(data_cache,'noisy','val'),os.path.join(results_dir,name,'audios','fake_B'),use_gender=False)
        avg_mssl,std_mssl = mssl(os.path.join(data_cache,'noisy','val'),os.path.join(results_dir,name,'audios','fake_B'),use_gender=False)

        shutil.rmtree(os.path.join(results_dir,name,'audios','fake_B'))

        info['avg_val_lsd'].append(avg_lsd)
        info['std_val_lsd'].append(std_lsd)
        info['avg_val_mssl'].append(avg_mssl)
        info['std_val_mssl'].append(std_mssl)

        if avg_lsd<min_lsd:
            min_lsd=avg_lsd
            min_lsd_epoch=epoch
        
        if avg_mssl<min_mssl:
            min_mssl=avg_mssl
            min_mssl_epoch=epoch
        
    info['min_val_lsd'] = min_lsd
    info['min_val_mssl'] = min_mssl
    info['min_val_lsd_epoch'] = min_lsd_epoch
    info['min_val_mssl_epoch'] = min_mssl_epoch

    return info


def apsipa_exp(names,csv_path,sources, data_cache='/content/AttentionGAN-VC/data_cache',results_dir='/content/AttentionGAN-VC/results'):
     for name, source in zip(names,sources):
        print('#'*25)
        print(f'Training {name} with Data from {source}')
        shutil.copytree(os.path.join('/content/drive/MyDrive/APSIPA/Data_Sources',source),data_cache)
        run(f'python train.py --dataroot data_cache --name {name} --model attention_gan --dataset_mode audio --pool_size 50 --no_dropout --norm instance --lambda_A 10 --lambda_B 10 --lambda_identity 0.5 --load_size_h 128 --load_size_w 128 --crop_size 128 --preprocess resize --batch_size 4 --niter 200 --niter_decay 0 --gpu_ids 0 --display_id 0 --display_freq 100 --print_freq 100 --input_nc 1 --output_nc 1 --use_cycled_discriminators --use_mask --max_mask_len 50 --checkpoints_dir /content/drive/MyDrive/APSIPA/Results/checkpoints --no_html')
        
        
        log(csv_path, name,f'Training {name} with Dataet source {source} for 200 epochs with cycled_disc, WITHOUT phase and WITH mask. LambdaA & B 10 , lambda_identity 0.5',avg_lsd,std_lsd,avg_mssl,std_mssl)
        shutil.rmtree(data_cache)
        print(f'Finished experiment with {name}')
        print('#'*25)