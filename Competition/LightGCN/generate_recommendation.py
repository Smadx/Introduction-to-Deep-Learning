import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
import multiprocessing

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset
#saved_model_path = '/home/wangyiming/projects/LightGCN-PyTorch-master/code/checkpoints/lgn-my_dataset_2-3-64.pth.tar'
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load from {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w: SummaryWriter = SummaryWriter(
        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
    )
else:
    w = None
    world.cprint("not enable tensorflowboard")


def generate_recommendation(Recmodel, output_file, dataset, top_k=10):
    # 调用 Test 函数并输出性能指标
    test_results, topk_predictions = Procedure.Test(dataset, Recmodel, 0,w,0)
    print("Test Results:", test_results)

    # 将推荐结果写入文件
    with open(output_file, 'w') as f:
        for user_id, items in topk_predictions:
            item_list = " ".join(map(str, items[:top_k]))
            f.write(f"{user_id} {item_list}\n")
    
    print(f"Recommendations saved to {output_file}")

# 使用示例
output_file = '../data/my_dataset_2/recommendations_10k.txt'
generate_recommendation(Recmodel, output_file, dataset, top_k=10)

'''
cd code && python generate_recommendation.py --decay=1e-4 --lr=0.005 --layer=3 --seed=2020 --dataset="my_dataset_2" --topks="[20]" --recdim=64

'''




input_file = '../data/my_dataset_2/recommendations_10k.txt'
submission_file = '../data/my_dataset_2/submission.csv'
output_file = '../data/my_dataset_2/mysubmission.csv'

# 读取推荐结果并保存到一个字典中
recommendations = {}
with open(input_file, 'r') as infile:
    for line in infile:
        parts = line.strip().split()
        user_id = parts[0]
        item_ids = parts[1:]
        recommendations[user_id] = item_ids

# 读取 submission 文件并填充推荐结果
with open(submission_file, 'r') as subfile, open(output_file, 'w') as outfile:
    header = subfile.readline().strip()
    outfile.write(f"{header}\n")
    for line in subfile:
        user_id, _ = line.strip().split(',')
        if user_id in recommendations and recommendations[user_id]:
            item_id = recommendations[user_id].pop(0)
            outfile.write(f"{user_id},{item_id}\n")
        else:
            # 如果推荐结果不够，则填充一个默认值或保留原值
            outfile.write(line)

print(f"Submission file with recommendations saved to {output_file}")