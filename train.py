import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms
from simba_torch.main import Simba
from torch.utils.data.dataloader import default_collate
import torchvision.transforms.functional as TF

coco_root = "./data/train2017/"
annot_path = os.path.join(coco_root, "../annotations")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ),
])

train_annot_path = os.path.join(
    annot_path, "instances_train2017.json"
)
val_annot_path = os.path.join(annot_path, "instances_val2017.json")

train_dataset = CocoDetection(
    root=coco_root, annFile=train_annot_path, transform=transform
)

val_dataset = CocoDetection(
    root=coco_root, annFile=val_annot_path, transform=transform
)


def custom_collate_fn(batch):
    images, targets = zip(*batch)

    images = [TF.resize(img, (224, 224)) for img in images]
    images = torch.stack(images)

    targets_list = []
    for target_dict in zip(*targets):
        padded_targets = {}
        for k in target_dict[0].keys():
            values = [d[k] for d in target_dict]
            max_lens = [
                max(
                    len(v)
                    for v in value
                    if isinstance(v, (list, tuple))
                )
                for value in values
            ]
            print(max_len)
            max_len = max(max_lens)
            padded_values = []
            for v in values:
                if isinstance(v, (list, tuple)):
                    padded_value = [
                        torch.tensor(
                            inner_v + [0] * (max_len - len(inner_v))
                        )
                        for inner_v in v
                    ]
                    padded_value = torch.stack(padded_value)
                    if padded_value.ndim > 2:
                        padded_value = padded_value.squeeze(1)
                elif isinstance(v, float):
                    padded_value = v
                else:
                    raise TypeError(
                        f"Unexpected type {type(v)} for value {v}"
                    )
                padded_values.append(padded_value)
            padded_targets[k] = torch.cat(padded_values, dim=0)
        targets_list.append(padded_targets)

    return images, targets_list


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=custom_collate_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=custom_collate_fn,
)


cls_loss_fn = nn.CrossEntropyLoss()
bbox_loss_fn = nn.SmoothL1Loss()


def segmentation_loss_fn(outputs, targets):
    loss = F.binary_cross_entropy_with_logits(outputs, targets)
    return loss


def detection_loss(outputs, targets):
    cls_outputs = outputs["cls"]
    bbox_outputs = outputs["bbox"]
    seg_outputs = outputs["segmentation"]

    gt_cls = targets["labels"]
    gt_bbox = targets["boxes"]
    gt_seg = targets["masks"]

    cls_loss = cls_loss_fn(cls_outputs, gt_cls)

    bbox_loss = bbox_loss_fn(bbox_outputs, gt_bbox)

    seg_loss = segmentation_loss_fn(seg_outputs, gt_seg)

    total_loss = cls_loss + bbox_loss + seg_loss

    return total_loss


model = Simba(
    dim=64,
    dropout=0.1,
    d_state=64,
    d_conv=64,
    num_classes=80,
    depth=8,
    patch_size=16,
    image_size=224,
    channels=3,
)
model = model.to("cuda")
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 32

for epoch in range(num_epochs):
    train_loss = 0.0
    val_loss = 0.0

    model.train()
    for images, targets in train_loader:
        images = images.to("cuda")
        targets = [
            {k: v.to("cuda") for k, v in t.items()} for t in targets
        ]

        outputs = model(images)
        loss = detection_loss(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to("cuda")
            targets = [
                {k: v.to("cuda") for k, v in t.items()}
                for t in targets
            ]

            outputs = model(images)
            loss = detection_loss(outputs, targets)

            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    print(
        f"Epoch: {epoch+1}, Training Loss: {train_loss}, Validation"
        f" Loss: {val_loss}"
    )
