# from tqdm import tqdm
import numpy as np
np.set_printoptions(suppress=True)
import os, sys
import torch
# import torch.multiprocessing.spawn as spawn
import torch.multiprocessing as mp
import time
import subprocess
from pathlib import Path

def run_cmd(cmd):
    #!/usr/bin/python
    ## get subprocess module 
    
    ## call date command ##
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    
    ## Talk with date command i.e. read data from stdout and stderr. Store this info in tuple ##
    ## Interact with process: Send data to stdin. Read data from stdout and stderr, until end-of-file is reached.  ##
    ## Wait for process to terminate. The optional input argument should be a string to be sent to the child process, ##
    ## or None, if no data should be sent to the child.
    (output, err) = p.communicate()
    
    ## Wait for date to terminate. Get return returncode ##
    p_status = p.wait()
    # print "Command output : ", output
    # print "Command exit status/return code : ", p_status
    return (output.decode(), err, p_status)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_total', type=int, default=8, help='total num of gpus available')
# parser.add_argument('--workers_total', type=int, default=-1, help='total num of workers; must be dividable by gpu_total, i.e. workers_total/gpu_total jobs per GPU')
opt = parser.parse_args()

def run_one_gpu(i, gpu_total, split, result_queue):
    torch.cuda.is_available()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(i)
    print("pid={} count={}".format(i, torch.cuda.device_count()))
    # assert count == 1
    
    # process_result_list = []
    # cmd = 'python test.py --name HERO_MODEL_scannet \
    #         --output_base_path outputs \
    #         --config_file configs/models/hero_model.yaml \
    #         --load_weights_from_checkpoint weights/hero_model.ckpt \
    #         --data_config configs/data/scannet_default_test.yaml \
    #         --num_workers 16 \
    #         --fast_cost_volume \
    #         --cache_depths \
    #         --dump_depth_visualization \
    #         --run_fusion \
    #         --depth_fuser open3d \
    #         --fuse_color \
    #         --gpu_id_debug %d \
    #         --gpu_total_debug %d \
    #         --batch_size 2;'%(i, gpu_total)
    cmd = 'python test.py --name HERO_MODEL_scannet_trainval \
            --output_base_path outputs \
            --config_file configs/models/hero_model.yaml \
            --load_weights_from_checkpoint weights/hero_model.ckpt \
            --data_config configs/data/scannet_default_train.yaml \
            --num_workers 16 \
            --fast_cost_volume \
            --cache_depths \
            --gpu_id_debug %d \
            --gpu_total_debug %d \
            --batch_size 2;'%(i, gpu_total)
            

    _results = run_cmd(cmd)
    print(_results)
    # time.sleep(2)
    # print("pid={} count={} SLEEP DONE".format(i, torch.cuda.device_count()), cmd)
        
    # return process_result_list
    # result_queue.put((i, _results))

if __name__ == '__main__':
    
    # if opt.workers_total == -1:
    #     opt.workers_total = opt.gpu_total
    
    # for split in ['train', 'val']:
    for split in ['test']:
        
        scene_name_list_path = Path(f'data_splits/ScanNetv2/standard_split/scannetv2_{split}.txt')
        with open(scene_name_list_path, 'r') as f:
            scene_name_list = [_.strip() for _ in f.readlines()]

        tic = time.time()
        
        result_queue = mp.Queue()
        for rank in range(opt.gpu_total):
            mp.Process(target=run_one_gpu, args=(rank, opt.gpu_total, split, result_queue)).start()
        
        for _ in range(opt.gpu_total):
            temp_result = result_queue.get()
            print(_, temp_result)
            
        print('==== ...DONE. Took %.2f seconds'%(time.time() - tic))