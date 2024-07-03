import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import xml.etree.ElementTree as ET
from torchvision import transforms

class ARTSv2(Dataset):
    """
    Dataset for loading images and annotations.

    :param root_dir: str, root directory of the dataset.
    :param transforms: torchvision.transforms, data transformations to be applied.
    """
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_dir = os.path.join(root_dir, 'images')
        self.ann_dir = os.path.join(root_dir, 'annotations')
        self.image_paths = [os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) if f.endswith('.jpg')]

        # Load class names from labels.txt
        labels_file = os.path.join(root_dir, 'labels.txt')
        with open(labels_file, 'r') as file:
            self.class_names = file.read().splitlines()
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Parse the corresponding annotation file
        ann_path = os.path.join(self.ann_dir, os.path.splitext(os.path.basename(img_path))[0] + '.xml')
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")

        tree = ET.parse(ann_path)
        root = tree.getroot()

        labels = []
        boxes = []
        gps = []

        for obj in root.findall('object'):
            label = obj.find('name').text
            labels.append(self.class_to_idx[label])  # Convert class name to index

            bndbox = obj.find('bndbox')
            bbox = [
                int(bndbox.find('xmin').text),
                int(bndbox.find('ymin').text),
                int(bndbox.find('xmax').text),
                int(bndbox.find('ymax').text)
            ]
            boxes.append(bbox)

            location = obj.find('location')
            gps_coords = [
                float(location.find('latitude').text),
                float(location.find('longitude').text)
            ]
            gps.append(gps_coords)
        
        labels = torch.tensor(labels, dtype=torch.long)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        gps = torch.tensor(gps, dtype=torch.float32)

        targets = {
            "labels": labels,
            "boxes": boxes,
            "gps": gps
        }

        if self.transforms:
            image = self.transforms(image)
        
        return image, targets

def custom_collate_fn(batch):
    images = []
    labels = []
    boxes = []
    gps = []

    max_num_objects = max(len(item[1]['labels']) for item in batch)

    for item in batch:
        image = item[0]
        target = item[1]

        padded_labels = torch.zeros((max_num_objects,), dtype=torch.long)
        padded_boxes = torch.zeros((max_num_objects, 4), dtype=torch.float32)
        padded_gps = torch.zeros((max_num_objects, 2), dtype=torch.float32)

        num_objects = len(target['labels'])

        padded_labels[:num_objects] = target['labels']
        padded_boxes[:num_objects] = target['boxes']
        padded_gps[:num_objects] = target['gps']

        images.append(image)
        labels.append(padded_labels)
        boxes.append(padded_boxes)
        gps.append(padded_gps)

    images = torch.stack(images)
    labels = torch.stack(labels)
    boxes = torch.stack(boxes)
    gps = torch.stack(gps)
    
    targets = {
        'labels': labels,
        'boxes': boxes,
        'gps': gps
    }

    return images, targets

def get_dataloader(root_dir, batch_size=5, shuffle=True, num_workers=4):
    """
    Creates and returns a DataLoader for the custom dataset.

    :param root_dir: str, root directory of the dataset.
    :param batch_size: int, number of samples per batch.
    :param shuffle: bool, whether to shuffle the dataset.
    :param num_workers: int, number of subprocesses to use for data loading.
    :return: DataLoader, data loader for the custom dataset.
    """
    # Transformations
    transform = transforms.Compose([
        transforms.Resize((640, 360)),  # Resize images to 640x360 pixels
        transforms.ToTensor()
    ])
    
    dataset = ARTSv2(root_dir, transforms=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate_fn  # Add the custom collate function here
    )
    return dataloader
