from pathlib import Path
import json
from tqdm import tqdm

for split in ['train', 'val', 'test']:
# for split in ['test']:
    scene_keyframe_dict = {}
    keyframe_file_path = Path(f'data_splits/ScanNetv2/standard_split/{split}_eight_view_deepvmvs.txt')
    keyframe_json_path = Path('/data/ruizhu/ScanNet/%s_keyframes.json'%split)
    
    SCANNET_ROOT = Path('/data/ruizhu/ScanNet/extracted_simplerecon/scans') if split in ['train', 'val'] else Path('/data/ruizhu/ScanNet/extracted_simplerecon/scans_test')
    
    assert keyframe_file_path.exists()
    with open(keyframe_file_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip().split(' ') for line in lines]
    for line in lines:
        scene_name, frame_id = line[0], int(line[1])
        if scene_name in scene_keyframe_dict:
            # scene_keyframe_dict[scene_name].append('i%d'%frame_id)
            scene_keyframe_dict[scene_name].append(frame_id)
        else:
            # scene_keyframe_dict[scene_name] = ['i%d'%frame_id]
            scene_keyframe_dict[scene_name] = [frame_id]
    
    print('Checking if all keyframes exist...')
    for scene_name in tqdm(scene_keyframe_dict):
        scene_path = SCANNET_ROOT / scene_name
        assert scene_path.exists(), scene_path
        for frame_id in scene_keyframe_dict[scene_name]:
            # frame_id = int(frame_id_[1:])
            for modality in ['.pose.txt', '.color.jpg', '.depth.png']:
                frame_path = scene_path / 'sensor_data' / ('frame-%06d.%s'%(frame_id, modality[1:]))
                assert frame_path.exists(), frame_path
                
    scene_name_list_path = Path(f'data_splits/ScanNetv2/standard_split/scannetv2_{split}.txt')
    with open(scene_name_list_path, 'r') as f:
        scene_name_list = [_.strip() for _ in f.readlines()]
    
    json_dict = {}
    for scene_name in scene_name_list:
        frame_list = scene_keyframe_dict[scene_name]
        frame_list.sort()
        json_dict[scene_name] = ['i%d'%_ for _ in frame_list]
    
    with open(str(keyframe_json_path), 'w') as f:
        json.dump(json_dict, f, indent=4)
        
    print('Saved to %s'%keyframe_json_path)


        
