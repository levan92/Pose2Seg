from pathlib import Path
from collections import defaultdict

logs = [
    '/Users/levan/Workspace/Pose2Seg_OCP/vm_logs/pose2seg/release_base_2022-07-22_09:39:01/train_logs.txt',
    '/Users/levan/Workspace/Pose2Seg_OCP/vm_logs/pose2seg/release_base_2022-07-22_09:39:47/train_logs.txt',
    '/Users/levan/Workspace/Pose2Seg_OCP/vm_logs/pose2seg/release_base_2022-07-22_09:41:19/train_logs.txt',
    '/Users/levan/Workspace/Pose2Seg_OCP/vm_logs/pose2seg/release_base_2022-07-22_08:53:35/train_logs.txt',
    '/Users/levan/Workspace/Pose2Seg_OCP/vm_logs/pose2seg_ocp/release_base_2022-07-23_13:19:40/train_logs.txt',
    '/Users/levan/Workspace/Pose2Seg_OCP/vm_logs/pose2seg_ocp/release_base_2022-07-23_13:30:02/train_logs.txt',
    '/Users/levan/Workspace/Pose2Seg_OCP/vm_logs/pose2seg_ocp/release_base_2022-07-23_13:30:49/train_logs.txt'
]

res = {
    'epoch':[],
    'AP':[],
    'set':[],
    'run':[],
}

count = defaultdict(int)

for log in logs: 
    with open(log,'r') as f: 
        lines = f.readlines()
    set = Path(log).parent.parent.stem
    count[set] += 1
    run = f'{set}_{count[set]}'

    epoch = 0
    for line in lines: 
        if '[segm_score] OCHumanVal' in line: 
            epoch += 1
            val = float(line.strip().split(' OCHumanVal ')[-1].split()[0])
            res['epoch'].append(epoch)
            res['AP'].append(val)
            res['set'].append(set)
            res['run'].append(run)

from pprint import pprint

# pprint(res)

import json 

with open('pose2seg_res.json', 'w') as f:
    json.dump(res, f)