import glob
import os
import random

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets')


def _clip_image(image, **kwargs):
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def build_train_transform():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ElasticTransform(
                alpha=8,
                sigma=20,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=0.3,
            ),
            A.Lambda(image=_clip_image),
            A.RandomGamma(gamma_limit=(85, 115), p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.05,
                contrast_limit=0.1,
                p=0.3,
            ),
            A.GaussNoise(std_range=(0.01, 0.04), mean_range=(0.0, 0.0), p=0.25),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
        ],
        additional_targets={'fov_mask': 'mask'},
    )


def _read_binary_mask(path, strict=True):
    if path is None or not os.path.exists(path):
        if strict:
            raise FileNotFoundError(f'Missing required mask file: {path}')
        return None
    return (cv2.imread(path, 0) > 127).astype(np.float32)


def _apply_green_clahe(bgr_image, clahe):
    green = bgr_image[:, :, 1]
    return cv2.normalize(
        clahe.apply(green),
        None,
        0,
        1,
        cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )


def _pad_and_resize(array, target_size, is_image):
    if target_size is None:
        return array
    target_h, target_w = target_size
    height, width = array.shape[:2]
    pad_h = max(0, target_h - height)
    pad_w = max(0, target_w - width)
    if pad_h or pad_w:
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        array = cv2.copyMakeBorder(array, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    interpolation = cv2.INTER_LINEAR if is_image else cv2.INTER_NEAREST
    if array.shape[:2] != (target_h, target_w):
        array = cv2.resize(array, (target_w, target_h), interpolation=interpolation)
    return array


def _to_tensor(image, label_first, label_second, fov_mask, dataset_name, name, has_second_label):
    return {
        'image': torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0),
        'label': torch.from_numpy(np.ascontiguousarray(label_first)).unsqueeze(0),
        'label_first': torch.from_numpy(np.ascontiguousarray(label_first)).unsqueeze(0),
        'label_second': torch.from_numpy(np.ascontiguousarray(label_second)).unsqueeze(0),
        'mask': torch.from_numpy(np.ascontiguousarray(fov_mask)).unsqueeze(0),
        'dataset': dataset_name,
        'name': name,
        'has_label': torch.tensor(True, dtype=torch.bool),
        'has_second_label': torch.tensor(has_second_label, dtype=torch.bool),
    }


class DriveDataset(Dataset):
    def __init__(self, split, image_size=592, train_ratio=0.8, seed=42, transform=None):
        self.split = split
        self.image_size = (image_size, image_size)
        self.transform = transform
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.items = self._build_items(train_ratio, seed)

    def _build_items(self, train_ratio, seed):
        train_images = sorted(glob.glob(os.path.join(DATASET_ROOT, 'DRIVE', 'training', 'images', '*.tif')))
        shuffled = train_images[:]
        random.Random(seed).shuffle(shuffled)
        split_index = max(1, int(len(shuffled) * train_ratio))
        train_subset = set(shuffled[:split_index])
        val_subset = set(shuffled[split_index:])
        items = []

        if self.split in {'train', 'val'}:
            selected = train_subset if self.split == 'train' else val_subset
            for img_path in sorted(selected):
                image_id = os.path.basename(img_path)[:2]
                items.append(
                    {
                        'name': os.path.splitext(os.path.basename(img_path))[0],
                        'img': img_path,
                        'label_first': glob.glob(os.path.join(DATASET_ROOT, 'DRIVE', 'training', '1st_manual', f'*{image_id}*.*'))[0],
                        'label_second': None,
                        'mask': glob.glob(os.path.join(DATASET_ROOT, 'DRIVE', 'training', 'mask', f'*{image_id}*.*'))[0],
                    }
                )
            return items

        if self.split == 'drive_test':
            for img_path in sorted(glob.glob(os.path.join(DATASET_ROOT, 'DRIVE', 'test', 'images', '*.tif'))):
                image_id = os.path.basename(img_path)[:2]
                items.append(
                    {
                        'name': os.path.splitext(os.path.basename(img_path))[0],
                        'img': img_path,
                        'label_first': glob.glob(os.path.join(DATASET_ROOT, 'DRIVE', 'test', '1st_manual', f'*{image_id}*.*'))[0],
                        'label_second': glob.glob(os.path.join(DATASET_ROOT, 'DRIVE', 'test', '2nd_manual', f'*{image_id}*.*'))[0],
                        'mask': glob.glob(os.path.join(DATASET_ROOT, 'DRIVE', 'test', 'mask', f'*{image_id}*.*'))[0],
                    }
                )
            return items

        raise ValueError(f'Unsupported DRIVE split: {self.split}')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        image = cv2.imread(item['img'])
        image = _apply_green_clahe(image, self.clahe)
        label_first = _read_binary_mask(item['label_first'])
        label_second = _read_binary_mask(item['label_second'], strict=False)
        fov_mask = _read_binary_mask(item['mask'])

        image = _pad_and_resize(image, self.image_size, is_image=True)
        label_first = _pad_and_resize(label_first, self.image_size, is_image=False)
        label_second = _pad_and_resize(label_second if label_second is not None else np.zeros_like(label_first), self.image_size, is_image=False)
        fov_mask = _pad_and_resize(fov_mask, self.image_size, is_image=False)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=label_first, fov_mask=fov_mask)
            image = np.nan_to_num(np.clip(transformed['image'], 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
            label_first = transformed['mask']
            fov_mask = transformed['fov_mask']

        label_first = (label_first > 0.5).astype(np.float32)
        label_second = (label_second > 0.5).astype(np.float32)
        fov_mask = (fov_mask > 0.5).astype(np.float32)
        return _to_tensor(image, label_first, label_second, fov_mask, 'DRIVE', item['name'], item['label_second'] is not None)


class ChasePatchDataset(Dataset):
    def __init__(
        self,
        split='train',
        patch_size=256,
        transform=None,
        patches_per_image=8,
        min_fov_ratio=0.3,
        train_ratio=0.8,
        seed=42,
    ):
        if split not in {'train', 'val'}:
            raise ValueError(f'Unsupported CHASE patch split: {split}')
        self.split = split
        self.patch_size = patch_size
        self.transform = transform
        self.patches_per_image = patches_per_image
        self.min_fov_ratio = min_fov_ratio
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.items = self._build_items(train_ratio, seed)

    def _build_items(self, train_ratio, seed):
        all_images = sorted(glob.glob(os.path.join(DATASET_ROOT, 'CHASEDB1', '*.jpg')))[:20]
        shuffled = all_images[:]
        random.Random(seed).shuffle(shuffled)
        split_index = max(1, int(len(shuffled) * train_ratio))
        train_subset = set(shuffled[:split_index])
        val_subset = set(shuffled[split_index:])
        selected = train_subset if self.split == 'train' else val_subset
        items = []
        for img_path in sorted(selected):
            mask_path = img_path.replace('.jpg', '_mask.png')
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f'CHASE official FOV mask is required: {mask_path}')
            items.append(
                {
                    'name': os.path.splitext(os.path.basename(img_path))[0],
                    'img': img_path,
                    'label_first': img_path.replace('.jpg', '_1stHO.png'),
                    'label_second': img_path.replace('.jpg', '_2ndHO.png'),
                    'mask': mask_path,
                }
            )
        return items

    def __len__(self):
        return len(self.items) * self.patches_per_image

    def _sample_patch(self, image, label_first, label_second, fov_mask):
        height, width = image.shape[:2]
        patch = self.patch_size
        max_top = max(height - patch, 0)
        max_left = max(width - patch, 0)

        for _ in range(20):
            top = 0 if max_top == 0 else random.randint(0, max_top)
            left = 0 if max_left == 0 else random.randint(0, max_left)
            patch_mask = fov_mask[top:top + patch, left:left + patch]
            if patch_mask.shape != (patch, patch):
                continue
            if patch_mask.mean() >= self.min_fov_ratio:
                return (
                    image[top:top + patch, left:left + patch],
                    label_first[top:top + patch, left:left + patch],
                    label_second[top:top + patch, left:left + patch],
                    patch_mask,
                )

        top = max_top // 2
        left = max_left // 2
        return (
            image[top:top + patch, left:left + patch],
            label_first[top:top + patch, left:left + patch],
            label_second[top:top + patch, left:left + patch],
            fov_mask[top:top + patch, left:left + patch],
        )

    def __getitem__(self, index):
        item = self.items[index % len(self.items)]
        image = cv2.imread(item['img'])
        image = _apply_green_clahe(image, self.clahe)
        label_first = _read_binary_mask(item['label_first'])
        label_second = _read_binary_mask(item['label_second'])
        fov_mask = _read_binary_mask(item['mask'])

        image, label_first, label_second, fov_mask = self._sample_patch(image, label_first, label_second, fov_mask)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=label_first, fov_mask=fov_mask)
            image = np.nan_to_num(np.clip(transformed['image'], 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
            label_first = transformed['mask']
            fov_mask = transformed['fov_mask']

        label_first = (label_first > 0.5).astype(np.float32)
        label_second = (label_second > 0.5).astype(np.float32)
        fov_mask = (fov_mask > 0.5).astype(np.float32)
        return _to_tensor(image, label_first, label_second, fov_mask, 'CHASEDB1', item['name'], True)


class ChaseFullImageDataset(Dataset):
    def __init__(self, split, train_ratio=0.8, seed=42):
        if split not in {'train', 'val', 'train_eval', 'test'}:
            raise ValueError(f'Unsupported CHASE split: {split}')
        self.split = split
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.items = self._build_items(train_ratio, seed)

    def _build_items(self, train_ratio, seed):
        all_images = sorted(glob.glob(os.path.join(DATASET_ROOT, 'CHASEDB1', '*.jpg')))
        train_eval_images = all_images[:20]
        test_images = all_images[20:]
        shuffled = train_eval_images[:]
        random.Random(seed).shuffle(shuffled)
        split_index = max(1, int(len(shuffled) * train_ratio))
        train_subset = set(shuffled[:split_index])
        val_subset = set(shuffled[split_index:])

        if self.split == 'train_eval':
            selected = train_eval_images
        elif self.split == 'train':
            selected = sorted(train_subset)
        elif self.split == 'val':
            selected = sorted(val_subset)
        else:
            selected = test_images
        items = []
        for img_path in selected:
            mask_path = img_path.replace('.jpg', '_mask.png')
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f'CHASE official FOV mask is required: {mask_path}')
            items.append(
                {
                    'name': os.path.splitext(os.path.basename(img_path))[0],
                    'img': img_path,
                    'label_first': img_path.replace('.jpg', '_1stHO.png'),
                    'label_second': img_path.replace('.jpg', '_2ndHO.png'),
                    'mask': mask_path,
                }
            )
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        image = cv2.imread(item['img'])
        image = _apply_green_clahe(image, self.clahe)
        label_first = _read_binary_mask(item['label_first'])
        label_second = _read_binary_mask(item['label_second'])
        fov_mask = _read_binary_mask(item['mask'])
        return _to_tensor(image, label_first, label_second, fov_mask, 'CHASEDB1', item['name'], True)
