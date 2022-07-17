import pandas as pd
import numpy as np
import os
import glob
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import albumentations as A

LABELS = {"ground": 0,
          "stomach": 1,
          "large_bowel": 2,
          "small_bowel": 3}

COLORS = {0: np.array([0, 0, 0]),
          1: np.array([0, 255, 0]),
          2: np.array([255, 255, 0]),
          3: np.array([255, 150, 150])}


def case_split_func(id: str):
    return id.split("_")[0]


def split_rle(codes: str):
    rles = codes.split(" ")
    rles = [[rles[idx * 2], rles[(idx * 2) + 1]] for idx in range(len(rles) // 2)]
    return np.array(rles).astype(int)


def image_file(id: str):
    splited_file = id.split("_")
    file_path = os.path.join("train", splited_file[0], f"{splited_file[0]}_{splited_file[1]}", "scans",
                             f"{splited_file[-2]}_{splited_file[-1]}*.png")
    file_path = glob.glob(file_path)[0]
    assert os.path.exists(file_path)
    return file_path


def to_list(name: str):
    return [name]


def extract_shape(path: str):
    width = int(path.split("\\")[-1].split("_")[2])
    height = int(path.split("\\")[-1].split("_")[3])
    return (height, width)


def read_dataset_csv(csv_file):
    df = pd.read_csv(csv_file)
    df.dropna(axis=0, inplace=True)
    df.reset_index(inplace=True, drop=True)
    df["case"] = df["id"].map(case_split_func)
    df["segmentation"] = df["segmentation"].map(split_rle)
    df["image_path"] = df["id"].map(image_file)
    df["class"] = df["class"].map(to_list)
    df["segmentation"] = df["segmentation"].map(to_list)
    df["shape"] = df["image_path"].map(extract_shape)


    duplicated_df = df[df.id.duplicated()]
    duplicated_df.reset_index(drop=True, inplace=True)

    deleted_dup_df = df[~df.id.duplicated()]
    deleted_dup_df.reset_index(drop=True, inplace=True)

    for idx in range(len(duplicated_df)):
        duplicate_data = duplicated_df.iloc[idx]
        duplicate_data_id = duplicate_data.id
        data = deleted_dup_df[deleted_dup_df.id == duplicate_data_id]
        data["class"].values[0] += duplicate_data["class"]
        data["segmentation"].values[0] += duplicate_data["segmentation"]

    return deleted_dup_df


def rle2mask(item: str, shape=(266, 266)):
    targets = []
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for rle, cls in zip(item["segmentation"], item["class"]):
        starts, lengths = rle.T
        starts -= 1
        ends = starts + lengths
        for lo, hi in zip(starts, ends):
            img[lo:hi] = LABELS[cls]
    return img.reshape(shape)


def visual_mask(mask):
    new_mask = np.zeros(mask.shape, dtype=np.uint8)
    new_mask = np.tile(new_mask[..., np.newaxis], [1, 1, 3])
    new_mask[np.where(mask == 0)] = COLORS[0]
    new_mask[np.where(mask == 1)] = COLORS[1]
    new_mask[np.where(mask == 2)] = COLORS[2]
    new_mask[np.where(mask == 3)] = COLORS[3]
    return new_mask / new_mask.max()


def plot_img_mask(img, mask):
    ploted = mask * 0.5 + img * 0.5
    from matplotlib.patches import Rectangle
    plt.imshow(ploted, cmap='bone')
    handles = [Rectangle((0,0),1,1, color=_c /255) for _c in list(COLORS.values())[1:]]
    labels = list(LABELS.keys())[1:]
    plt.legend(handles,labels, loc=2)
    plt.axis('off')
    plt.show()


class UWMDataset(Dataset):
    def __init__(self, dataframe, transforms, num_classes):
        self.dataframe = dataframe
        self.transforms = transforms
        self.num_classes = num_classes

    def __getitem__(self, item):
        img = self.dataframe.image_path.iloc[item]
        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        img = np.tile(img[..., np.newaxis], [1, 1, 3])
        mask = rle2mask(self.dataframe.iloc[item], shape=self.dataframe["shape"].iloc[item])
        transformed = self.transforms(image=img, mask=mask)
        transformed_image = transformed['image']
        transformed_image = transformed_image / transformed_image.max()
        transformed_mask = transformed['mask']
        transformed_image = torch.tensor(transformed_image).permute(2, 0, 1)
        transformed_mask = torch.tensor(transformed_mask).long()
        transformed_mask = F.one_hot(transformed_mask, num_classes=self.num_classes).permute(2, 0, 1)
        return transformed_image, transformed_mask

    def __len__(self):
        return len(self.dataframe)


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
    return batched_imgs


def collate_fn(batch):
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets

# train_transforms = A.Compose([
#     A.Resize()
# ])

if __name__ == "__main__":
    dataset = read_dataset_csv("train.csv")
    print("Load Dataset")
    train_transform = A.Compose([
        A.Resize(height=224, width=224),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ElasticTransform(p=0.5),
    ])
    test_transform = A.Compose([
        A.Resize(height=224, width=224),
    ])
    train_ds = UWMDataset(dataframe=dataset[:-590], transforms=train_transform, num_classes=4)
    test_ds = UWMDataset(dataframe=dataset[-590:], transforms=test_transform, num_classes=4)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=2,
        shuffle=True,
    )
    batch_train = next(iter(train_loader))
    batch_test = next(iter(test_loader))
