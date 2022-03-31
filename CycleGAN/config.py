import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
BATCH_SIZE = 1
# LEARNING_RATE = 1e-5
LEARNING_RATE = 2e-4
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
# NUM_EPOCHS = 10
NUM_EPOCHS = 100
LOAD_MODEL = False
# test 시에는 Load_model True로 변경할것
SAVE_MODEL = True
CHECKPOINT_GEN_H = "./checkpoint/GenoratorH.pt"
CHECKPOINT_GEN_Z = "./checkpoint/GenoratorZ.pt"
CHECKPOINT_CRITIC_H = "./checkpoint/CriticH.pt"
CHECKPOINT_CRITIC_Z = "./checkpoint/CriticZ.pt"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)
