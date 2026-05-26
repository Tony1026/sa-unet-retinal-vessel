import glob
import hashlib
import os
import random

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets'))


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
        additional_targets={'fov_mask': 'mask', 'label_second': 'mask'},
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


def _preprocess_image(bgr_image, clahe, input_mode):
    if input_mode == 'green_clahe':
        return _apply_green_clahe(bgr_image, clahe)
    if input_mode == 'green':
        return bgr_image[:, :, 1].astype(np.float32) / 255.0
    if input_mode == 'rgb':
        return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    raise ValueError(f'Unsupported input mode: {input_mode}')


def input_channels_for_mode(input_mode):
    return 3 if input_mode == 'rgb' else 1


def resolve_data_root(data_root=None):
    return os.path.abspath(data_root or os.environ.get('RETINA_DATA_ROOT', DEFAULT_DATASET_ROOT))


def _valid_region_after_pad(original_shape, target_size):
    if target_size is None:
        height, width = original_shape
        return (0, 0, height, width)

    target_h, target_w = target_size
    height, width = original_shape
    if target_h < height or target_w < width:
        return (0, 0, target_h, target_w)

    top = (target_h - height) // 2
    left = (target_w - width) // 2
    return (top, left, height, width)


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


def _stable_annotation_rng(name):
    digest = hashlib.sha256(str(name).encode('utf-8')).digest()
    seed = int.from_bytes(digest[:8], byteorder='little') % (2**32)
    return random.Random(seed), np.random.default_rng(seed)


def _random_array(np_rng, shape):
    return np_rng.random(shape)


def _simulate_annotation_bias(label, fov_mask, rng=None, np_rng=None):
    rng = rng or random
    np_rng = np_rng or np.random
    label_u8 = (label > 0.5).astype(np.uint8)
    kernel_size = rng.choice([3, 3, 5])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    op = rng.choice(['dilate', 'erode', 'open', 'close', 'soft_threshold'])

    if op == 'dilate':
        biased = cv2.dilate(label_u8, kernel, iterations=1)
    elif op == 'erode':
        biased = cv2.erode(label_u8, kernel, iterations=1)
    elif op == 'open':
        biased = cv2.morphologyEx(label_u8, cv2.MORPH_OPEN, kernel)
    elif op == 'close':
        biased = cv2.morphologyEx(label_u8, cv2.MORPH_CLOSE, kernel)
    else:
        blurred = cv2.GaussianBlur(label_u8.astype(np.float32), (5, 5), 0)
        biased = (blurred > rng.uniform(0.42, 0.58)).astype(np.uint8)

    if rng.random() < 0.35:
        boundary = cv2.dilate(label_u8, kernel, iterations=1) - cv2.erode(label_u8, kernel, iterations=1)
        boundary_noise = (_random_array(np_rng, label_u8.shape) > rng.uniform(0.35, 0.65)).astype(np.uint8)
        biased = np.where(boundary > 0, np.maximum(biased, boundary_noise * label_u8), biased)

    return (biased.astype(np.float32) * (fov_mask > 0.5).astype(np.float32))


def _simulate_annotation_bias_for_sample(label, fov_mask, name, split):
    rng, np_rng = _stable_annotation_rng(f'{split}:{name}:boundary-bias')
    return _simulate_annotation_bias(label, fov_mask, rng=rng, np_rng=np_rng)


def _image_to_tensor(image):
    if image.ndim == 2:
        return torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


def _to_tensor(image, label_first, label_second, fov_mask, dataset_name, name, has_second_label, valid_region=None):
    height, width = label_first.shape[:2]
    if valid_region is None:
        valid_region = (0, 0, height, width)
    return {
        'image': _image_to_tensor(image),
        'label': torch.from_numpy(np.ascontiguousarray(label_first)).unsqueeze(0),
        'label_first': torch.from_numpy(np.ascontiguousarray(label_first)).unsqueeze(0),
        'label_second': torch.from_numpy(np.ascontiguousarray(label_second)).unsqueeze(0),
        'mask': torch.from_numpy(np.ascontiguousarray(fov_mask)).unsqueeze(0),
        'valid_region': torch.tensor(valid_region, dtype=torch.long),
        'dataset': dataset_name,
        'name': name,
        'has_label': torch.tensor(True, dtype=torch.bool),
        'has_second_label': torch.tensor(has_second_label, dtype=torch.bool),
    }


class DriveDataset(Dataset):
    def __init__(
        self,
        split,
        image_size=592,
        train_ratio=0.8,
        seed=42,
        transform=None,
        input_mode='green_clahe',
        data_root=None,
        label_bias_mode='none',
    ):
        self.split = split
        self.image_size = (image_size, image_size)
        self.transform = transform
        self.input_mode = input_mode
        self.label_bias_mode = label_bias_mode
        self.data_root = resolve_data_root(data_root)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.items = self._build_items(train_ratio, seed)

    def _build_items(self, train_ratio, seed):
        train_images = sorted(glob.glob(os.path.join(self.data_root, 'DRIVE', 'training', 'images', '*.tif')))
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
                        'label_first': glob.glob(os.path.join(self.data_root, 'DRIVE', 'training', '1st_manual', f'*{image_id}*.*'))[0],
                        'label_second': None,
                        'mask': glob.glob(os.path.join(self.data_root, 'DRIVE', 'training', 'mask', f'*{image_id}*.*'))[0],
                    }
                )
            return items

        if self.split == 'drive_test':
            for img_path in sorted(glob.glob(os.path.join(self.data_root, 'DRIVE', 'test', 'images', '*.tif'))):
                image_id = os.path.basename(img_path)[:2]
                items.append(
                    {
                        'name': os.path.splitext(os.path.basename(img_path))[0],
                        'img': img_path,
                        'label_first': glob.glob(os.path.join(self.data_root, 'DRIVE', 'test', '1st_manual', f'*{image_id}*.*'))[0],
                        'label_second': glob.glob(os.path.join(self.data_root, 'DRIVE', 'test', '2nd_manual', f'*{image_id}*.*'))[0],
                        'mask': glob.glob(os.path.join(self.data_root, 'DRIVE', 'test', 'mask', f'*{image_id}*.*'))[0],
                    }
                )
            return items

        raise ValueError(f'Unsupported DRIVE split: {self.split}')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        bgr_image = cv2.imread(item['img'])
        original_shape = bgr_image.shape[:2]
        image = _preprocess_image(bgr_image, self.clahe, self.input_mode)
        label_first = _read_binary_mask(item['label_first'])
        label_second = _read_binary_mask(item['label_second'], strict=False)
        fov_mask = _read_binary_mask(item['mask'])
        valid_region = _valid_region_after_pad(original_shape, self.image_size)

        image = _pad_and_resize(image, self.image_size, is_image=True)
        label_first = _pad_and_resize(label_first, self.image_size, is_image=False)
        label_second = _pad_and_resize(label_second if label_second is not None else np.zeros_like(label_first), self.image_size, is_image=False)
        fov_mask = _pad_and_resize(fov_mask, self.image_size, is_image=False)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=label_first, label_second=label_second, fov_mask=fov_mask)
            image = np.nan_to_num(np.clip(transformed['image'], 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
            label_first = transformed['mask']
            label_second = transformed['label_second']
            fov_mask = transformed['fov_mask']

        label_first = (label_first > 0.5).astype(np.float32)
        label_second = (label_second > 0.5).astype(np.float32)
        fov_mask = (fov_mask > 0.5).astype(np.float32)
        has_second_label = item['label_second'] is not None
        if self.label_bias_mode == 'second':
            label_second = _simulate_annotation_bias(label_first, fov_mask)
            has_second_label = True
        elif self.split == 'train' and self.label_bias_mode == 'random_primary':
            label_first = _simulate_annotation_bias(label_first, fov_mask)
        elif self.label_bias_mode == 'random_primary':
            label_second = _simulate_annotation_bias_for_sample(label_first, fov_mask, item['name'], self.split)
            has_second_label = True
        return _to_tensor(image, label_first, label_second, fov_mask, 'DRIVE', item['name'], has_second_label, valid_region)


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
        input_mode='green_clahe',
        data_root=None,
        label_bias_mode='none',
    ):
        if split not in {'train', 'val'}:
            raise ValueError(f'Unsupported CHASE patch split: {split}')
        self.split = split
        self.patch_size = patch_size
        self.transform = transform
        self.patches_per_image = patches_per_image
        self.min_fov_ratio = min_fov_ratio
        self.input_mode = input_mode
        self.label_bias_mode = label_bias_mode
        self.data_root = resolve_data_root(data_root)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.items = self._build_items(train_ratio, seed)

    def _build_items(self, train_ratio, seed):
        all_images = sorted(glob.glob(os.path.join(self.data_root, 'CHASEDB1', '*.jpg')))[:20]
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
        image = _preprocess_image(cv2.imread(item['img']), self.clahe, self.input_mode)
        label_first = _read_binary_mask(item['label_first'])
        label_second = _read_binary_mask(item['label_second'])
        fov_mask = _read_binary_mask(item['mask'])

        image, label_first, label_second, fov_mask = self._sample_patch(image, label_first, label_second, fov_mask)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=label_first, label_second=label_second, fov_mask=fov_mask)
            image = np.nan_to_num(np.clip(transformed['image'], 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
            label_first = transformed['mask']
            label_second = transformed['label_second']
            fov_mask = transformed['fov_mask']

        label_first = (label_first > 0.5).astype(np.float32)
        label_second = (label_second > 0.5).astype(np.float32)
        fov_mask = (fov_mask > 0.5).astype(np.float32)
        if self.split == 'train' and self.label_bias_mode == 'second':
            label_second = _simulate_annotation_bias(label_first, fov_mask)
        elif self.split == 'train' and self.label_bias_mode == 'random_primary':
            label_first = _simulate_annotation_bias(label_first, fov_mask)
        return _to_tensor(image, label_first, label_second, fov_mask, 'CHASEDB1', item['name'], True)


class ChaseFullImageDataset(Dataset):
    def __init__(self, split, train_ratio=0.8, seed=42, input_mode='green_clahe', data_root=None, label_bias_mode='none'):
        if split not in {'train', 'val', 'train_eval', 'test'}:
            raise ValueError(f'Unsupported CHASE split: {split}')
        self.split = split
        self.input_mode = input_mode
        self.label_bias_mode = label_bias_mode
        self.data_root = resolve_data_root(data_root)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.items = self._build_items(train_ratio, seed)

    def _build_items(self, train_ratio, seed):
        all_images = sorted(glob.glob(os.path.join(self.data_root, 'CHASEDB1', '*.jpg')))
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
        image = _preprocess_image(cv2.imread(item['img']), self.clahe, self.input_mode)
        label_first = (_read_binary_mask(item['label_first']) > 0.5).astype(np.float32)
        label_second = (_read_binary_mask(item['label_second']) > 0.5).astype(np.float32)
        fov_mask = (_read_binary_mask(item['mask']) > 0.5).astype(np.float32)
        if self.label_bias_mode in {'second', 'random_primary'}:
            label_second = _simulate_annotation_bias_for_sample(label_first, fov_mask, item['name'], self.split)
        return _to_tensor(image, label_first, label_second, fov_mask, 'CHASEDB1', item['name'], True)
