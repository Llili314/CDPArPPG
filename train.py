# train_sfda_physformer.py
import os, argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from model import ViT_ST_ST_Compact3_TDC_gra_sharp
from load_data import PKL_TargetOnly
from data_transform import (
    Normaliztion_TargetOnly, RandomHorizontalFlip_TargetOnly, ToTensor_TargetOnly
)
from utils import update_ema, z_norm, NegPearson, compute_inter_time_graph_consistency,compute_intra_graph_consistency
from period_detection.find_period import find_period

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, default=r'your pretrained model directory', help="path to .pth (state_dict)")
    parser.add_argument("--target_video_dir", type=str, default=r'your dataset directory', help='RGB video pkl directory')
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--ema_m", type=float, default=0.99)
    parser.add_argument("--lambda_graph", type=float, default=1)
    parser.add_argument("--gra_sharp", type=float, default=2.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="sfda_ckpt")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # 1) 构建 student / teacher（结构要和你预训练一致）
    student = ViT_ST_ST_Compact3_TDC_gra_sharp(
        image_size=(128, 128, 128),
        patches=(4, 4, 4),
        dim=96,
        ff_dim=144,
        num_heads=4,
        num_layers=8,
        dropout_rate=0.1,
        theta=0.7
    ).to(device)

    teacher = ViT_ST_ST_Compact3_TDC_gra_sharp(
        image_size=(128, 128, 128),
        patches=(4, 4, 4),
        dim=96,
        ff_dim=144,
        num_heads=4,
        num_layers=8,
        dropout_rate=0.1,
        theta=0.7
    ).to(device)

    ckpt = torch.load(args.pretrained, map_location=device)
    student.load_state_dict(ckpt)
    teacher.load_state_dict(ckpt)
    #teacher.eval() + requires_grad=False = Teacher 是一个“稳定的、慢更新的监督器”
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    # 2) 目标域 dataloader（无标签）
    ds = PKL_TargetOnly(
        video_dir=args.target_video_dir,
        fps_value=args.fps,
        transform=transforms.Compose([
            Normaliztion_TargetOnly(),
            RandomHorizontalFlip_TargetOnly(),
            ToTensor_TargetOnly()
        ])
    )
    loader = DataLoader(
        ds,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    optimizer = optim.Adam(student.parameters(), lr=args.lr, weight_decay=5e-5)
    crit_wave = NegPearson()

    dt_minutes = (1.0 / args.fps) / 60.0   # 每帧间隔（分钟）


    # 3) 训练（SFDA）
    for epoch in range(args.epochs):
        student.train()

        total_loss = 0.0
        total_wave = 0.0
        total_graph = 0.0
        n = 0

        for batch in loader:
            x = batch["video_x"].to(device)  # [B,3,T,H,W]（通道前移来自 ToTensor 逻辑）
            # 你的原 ToTensor 是 (3,T,H,W) :contentReference[oaicite:9]{index=9}

            # ---- teacher forward (no grad) ----
            with torch.no_grad():
                r_t, tok_t, _, _, amp_t = teacher(x, args.gra_sharp,return_att=True)
                # tok_t，B × P × C} = (4, 512, 96)，表示： 4 个样本 × 512 个时空 token × 每个 token 是 96 维特征


            r_value_list = []
            for b in range(r_t.shape[0]):
                res,r_value = find_period_torch(r_t[b].detach(),dt_minutes=dt_minutes,minimum_number_of_datapoints_for_correlation_test=30,use_pvalue_approx=False)   #如果正常返回model-data，数据类型按下面处理，若检测不到任何周期特性，则返回数据本身，数据类型格式不一样，需调整
                r_value_list.append(r_value.squeeze(-1) if r_value.dim()==2 else r_value)
            r_t_value = torch.stack(r_value_list, dim=0)  # (B, T)



            # ---- student forward ----
            r_s, tok_s, _, _ , amp_s = student(x, args.gra_sharp,return_att=True)

            # ---- wave consistency ----
            rs = z_norm(r_s)
            rt = z_norm(r_t_value.detach())
            loss_wave = crit_wave(rs, rt.detach())

            # ---- token-graph consistency ----
            loss_inter_graph = compute_inter_time_graph_consistency(tok_s, tok_t)
            loss_intra_graph = compute_intra_graph_consistency(tok_s, tok_t)

            loss = loss_wave + args.lambda_graph * loss_inter_graph + loss_intra_graph

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #update_ema(teacher, student, args.ema_m)

            bsz = x.size(0)
            total_loss += loss.item() * bsz
            total_wave += loss_wave.item() * bsz
            total_graph += (loss_inter_graph.item()+loss_intra_graph.item()) * bsz
            n += bsz

        print(
            f"[Epoch {epoch+1}/{args.epochs}] "
            f"total={total_loss/n:.4f} | wave={total_wave/n:.4f} | graph={total_graph/n:.4f}"
        )

        torch.save(student.state_dict(), os.path.join(args.save_dir, f"student_sfda_epoch_{epoch:03d}.pth"))
        #torch.save(teacher.state_dict(), os.path.join(args.save_dir, f"teacher_sfda_epoch_{epoch:03d}.pth"))

if __name__ == "__main__":
    main()
