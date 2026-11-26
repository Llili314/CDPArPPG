# data_transform_target_only.py
import numpy as np
import torch
import random
import cv2

class Normaliztion_TargetOnly(object):
    def __call__(self, sample):
        video_x, frame_rate = sample["video_x"], sample["frame_rate"]
        video_x = (video_x - 127.5) / 128.0
        return {"video_x": video_x, "frame_rate": frame_rate}

class RandomHorizontalFlip_TargetOnly(object):
    def __call__(self, sample):
        video_x, frame_rate = sample["video_x"], sample["frame_rate"]
        T, h, w = video_x.shape[0], video_x.shape[1], video_x.shape[2]
        if random.random() < 0.5:
            out = np.zeros((T, h, w, 3), dtype=video_x.dtype)
            for i in range(T):
                out[i] = cv2.flip(video_x[i], 1)
            video_x = out
        return {"video_x": video_x, "frame_rate": frame_rate}

class ToTensor_TargetOnly(object):
    def __call__(self, sample):
        video_x = sample["video_x"]              # (T,H,W,3)
        frame_rate = sample["frame_rate"]
        video_x = np.transpose(video_x, (3, 0, 1, 2))  # (3,T,H,W)
        return {
            "video_x": torch.from_numpy(video_x).float(),
            "frame_rate": torch.tensor(frame_rate, dtype=torch.float32),
        }
