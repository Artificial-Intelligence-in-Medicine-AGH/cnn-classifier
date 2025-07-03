import albumentations as A
from albumentations.pytorch import ToTensorV2

val_transform = A.Compose([
    A.Normalize(mean=[0.5], std=[0.5]),
    ToTensorV2(),
])

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.3),
    A.Rotate(limit=10, p=0.5),
    A.RGBShift(p=0.3),
    A.GaussianBlur(blur_limit=(1, 3), p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.RandomGamma(p=0.3),
    A.SaltAndPepper(p=0.5),
    A.Normalize(mean=[0.5], std=[0.5]),
    ToTensorV2(),
])
