import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 参数设置
npop = 300  # 种群规模
sigma = 0.1  # 噪声标准差
alpha = 0.008  # 学习率
epsi = 0.031  # 扰动的最大范围
num_steps = 400  # 攻击的迭代次数

def generate_noise(shape, sigma):
    return torch.randn(shape).cuda() * sigma

def attack_point_cloud(model, point_cloud, target):
    B, _, N = point_cloud.shape  # B: batch size, N: number of points

    # 初始化扰动
    modify = torch.randn(B, 3, N).cuda() * 0.001

    for step in range(num_steps):
        # 生成噪声并添加到原始点云
        Nsample = generate_noise((npop, 3, N), sigma)
        modify_try = modify.repeat(npop, 1, 1) + sigma * Nsample
        perturbed_point_cloud = point_cloud.unsqueeze(0).repeat(npop, 1, 1, 1) + modify_try

        # 限制扰动的范围
        dist = perturbed_point_cloud - point_cloud.unsqueeze(0).repeat(npop, 1, 1, 1)
        clipdist = torch.clamp(dist, -epsi, epsi)
        clipinput = point_cloud.unsqueeze(0).repeat(npop, 1, 1, 1) + clipdist

        # 计算模型输出
        logits = model(clipinput.transpose(2, 3))  # 转置使得维度符合模型输入 (B, N, 3)
        target_onehot = torch.zeros(npop, logits.shape[-1]).cuda()
        target_onehot[:, target] = 1

        real = torch.log((target_onehot * logits).sum(1) + 1e-30)
        other = torch.log(((1. - target_onehot) * logits).max(1)[0] + 1e-30)
        loss = torch.clamp(real - other, min=0.)

        # 奖励和更新
        Reward = -0.5 * loss
        A = (Reward - Reward.mean()) / (Reward.std() + 1e-7)
        modify = modify + (alpha / (npop * sigma)) * torch.matmul(Nsample.view(npop, -1).T, A).view(3, N)

    return modify

# 假设 model 是预训练的点云分类模型，例如 PointNet
# 加载并设置模型到GPU
model = model.cuda()
model.eval()

# 假设 point_cloud 是要攻击的输入点云， target 是目标类别
point_cloud = point_cloud.cuda()  # (B, 3, N)
target = target.cuda()  # (B,)

# 执行攻击
perturbation = attack_point_cloud(model, point_cloud, target)

# 应用扰动并获得最终对抗性点云
adv_point_cloud = point_cloud + perturbation

# 评估攻击效果
with torch.no_grad():
    logits = model(adv_point_cloud.transpose(1, 2))
    pred = logits.argmax(dim=-1)
    success = pred != target
    print(f"Attack success: {success.item()}")
