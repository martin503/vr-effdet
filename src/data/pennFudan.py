import torch
import torchvision.transforms as T
import numpy as np
import copy
from PIL import Image
from ..helpers import seed_all


class PennFudanDataset(torch.utils.data.Dataset):
    FILLER = 69420.2137
    SEED_VAL = 2147483647
    MAX_NUM_PPL = 13
    NUM_CLASSES = 2

    def __init__(self, imgs, masks, threshold, transforms=None):
        """PennFudanPed dataset for semantic segmentation with optional transformations

        Args:
            imgs (List[str]): paths to images
            masks (List[str]): paths to masks
            threshold (float): minimal area of object to be classified,
            specified in percentage of image coverage
            transforms (torch.nn.Module, optional): transformations that will be applied to both masks and images
        """
        self.transforms = transforms
        self.imgs = imgs
        self.threshold = threshold
        self.masks = masks

    def __getitem__(self, idx):
        # load images and masks
        img = Image.open(self.imgs[idx]).convert('RGB')
        mask = Image.open(self.masks[idx])
        mask = np.array(mask)

        # Instances are encoded as different colors
        obj_ids = np.unique(mask)
        # First id is the background, so remove it
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]

        # Apply transformations to imgs and masks
        if self.transforms is not None:
            seed = np.random.randint(PennFudanDataset.SEED_VAL)
            seed_all(seed)
            img = self.transforms(img)
            to_np = T.Compose([T.ToPILImage(), np.array])
            new_masks = []
            for mask in masks:
                # Seed so we get the same transformation for image and for boxes
                seed_all(seed)
                new_mask = self.__reduce_colors(
                    np.sum(to_np(self.transforms(Image.fromarray(mask).convert('RGB'))), axis=2))
                if new_mask is not None:
                    new_masks.append(new_mask)
            if len(new_masks) == 0:
                return self.__getitem__(idx + 1)
            masks = torch.as_tensor(np.array(new_masks))

        # Get bounding boxes coordinates for each mask
        boxes = self.__get_boxes(masks)

        # Get masks
        masks = T.Resize(masks.shape[2] // 2)(masks)
        masks = torch.sum(masks, dim=0)
        masks.apply_(lambda x: 0 if x < 0.7 else 1)
        masks.unsqueeze_(dim=0)
        masks = masks.type(torch.float32)

        # Get edge mask for each mask
        edge_masks = self.__get_edge_masks(masks)

        return img, masks, edge_masks, boxes

    def __len__(self):
        return len(self.imgs)

    def __reduce_colors(self, mask):
        val, counts = np.unique(mask, return_counts=True)
        if len(counts) == 1:
            return None
        m1, m2 = (-counts).argsort()[:2]
        v1, v2 = val[m1], val[m2]
        reduce = np.vectorize(lambda x: 0 if abs(int(x) - int(v1)) < abs(int(x) - int(v2)) else 1)
        new_mask = reduce(mask).astype(np.uint8)
        val, counts = np.unique(new_mask, return_counts=True)
        if len(counts) == 1 or counts[1] / counts[0] < self.threshold:
            return None
        return new_mask

    def __normalize_box(self, box, H, W):
        return [box[0] / W, box[1] / H, box[2] / W, box[3] / H]

    def __get_boxes(self, masks):
        num_objs = len(masks)
        boxes = torch.full((PennFudanDataset.MAX_NUM_PPL, 4), PennFudanDataset.FILLER, dtype=torch.float32)
        for i in range(num_objs):
            pos = np.where(masks[i] == 1)
            x_min = np.min(pos[1])
            y_min = np.min(pos[0])
            x_max = (np.max(pos[1]))
            y_max = (np.max(pos[0]))
            boxes[i, :4] = torch.as_tensor((self.__normalize_box([x_min, y_min, x_max, y_max], *masks[i].shape)),
                                           dtype=torch.float32)
        return boxes

    def __get_edge_masks(self, masks):
        with torch.no_grad():
            conv = torch.nn.Conv2d(1, 4, kernel_size=3, padding=1, stride=1, bias=False)
            conv.weight = torch.nn.parameter.Parameter(torch.tensor([
                [[[0, 0, 0],
                  [-1, 0, 1],
                  [0, 0, 0]]],
                [[[0, -1, 0],
                  [0, 0, 0],
                  [0, 1, 0]]],
            ], dtype=torch.float32))
            edge_masks = conv(masks)
            edge_masks = torch.abs(edge_masks)
            edge_masks = torch.sum(edge_masks, dim=0)
            edge_masks.apply_(lambda x: 0 if x < 0.9 else 1)
            edge_masks.unsqueeze_(dim=0)
            return edge_masks

    @staticmethod
    def denormalize_boxes(boxes, H, W):
        boxes = copy.deepcopy(boxes)
        for i in range(len(boxes)):
            boxes[i][0] *= W
            boxes[i][1] *= H
            boxes[i][2] *= W
            boxes[i][3] *= H
        return boxes

    @staticmethod
    def get_boxes(target):
        num_boxes = 0
        for i in range(target.shape[0]):
            if PennFudanDataset.FILLER in target[i]:
                num_boxes = i
                break
        return copy.deepcopy(target[:num_boxes])
