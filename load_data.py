# Load_data_pkl_target_only.py
import os, joblib, numpy as np, cv2
from torch.utils.data import Dataset

class PKL_TargetOnly(Dataset):
    def __init__(self, video_dir, fps_value=20.0, transform=None):
        self.video_paths = sorted([
            os.path.join(video_dir, f)
            for f in os.listdir(video_dir) if f.endswith(".pkl")
        ])
        self.fps_value = float(fps_value)
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video = joblib.load(self.video_paths[idx])          # (T,224,224,3) or (T,128,128,3)
        video = np.asarray(video, dtype=np.float32)
        frame_rate = np.asarray(self.fps_value, dtype=np.float32)

        # resize -> 128x128（复用你现有逻辑）
        T, H, W, C = video.shape
        if (H, W) != (128, 128):
            new_video = np.zeros((T, 128, 128, C), dtype=np.float32)
            for t in range(T):
                new_video[t] = cv2.resize(video[t], (128, 128))
            video = new_video

        sample = {"video_x": video, "frame_rate": frame_rate}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
