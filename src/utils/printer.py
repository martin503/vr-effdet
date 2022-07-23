import matplotlib.pyplot as plt
import torch
import torchvision
import random


class Printer:
    WIDTH = 2

    @staticmethod
    def imshow(*imgs):
        _, axarr = plt.subplots(1, len(imgs), figsize=(10, 10), squeeze=False)
        for i, img in enumerate(imgs):
            axarr[0, i].imshow(img)
        plt.show()

    @staticmethod
    def imshow_random(*imgs, num_imgs_to_show=5):
        if len(imgs) > 0:
            to_show_id = random.randint(0, max(0, len(imgs) - num_imgs_to_show))
            Printer.imshow(*imgs[to_show_id:to_show_id + num_imgs_to_show])

    @staticmethod
    def imshow_with_masks_and_boxes(img_size, *imgs_masks_boxes):
        assert len(imgs_masks_boxes) % 4 == 0
        for i in range(len(imgs_masks_boxes) // 4):
            to_show = []
            if len(imgs_masks_boxes[4 * i + 3]) == 0:
                Printer.imshow((255 * imgs_masks_boxes[4 * i]).type(torch.uint8).permute(1, 2, 0))
                continue
            boxes = imgs_masks_boxes[4 * i + 3] * img_size
            img_with_box = torchvision.utils.draw_bounding_boxes((255 * imgs_masks_boxes[4 * i]).type(torch.uint8), boxes.type(torch.int), colors=(0, 255, 255), width=Printer.WIDTH)
            to_show.append(img_with_box.permute(1, 2, 0))
            for mask, edge_mask in zip(imgs_masks_boxes[4 * i + 1], imgs_masks_boxes[4 * i + 2]):
                to_show.append((mask).type(torch.uint8))
                to_show.append((edge_mask).type(torch.uint8))
            to_show = [img.cpu() for img in to_show]
            Printer.imshow(*to_show)
    
    @staticmethod
    def imshow_with_masks_and_boxes_random(img_size, *imgs, num_imgs_to_show=5):
        assert len(imgs) % 4 == 0
        if len(imgs) > 0:
            to_show_id = random.randint(0, max(0, len(imgs) // 4 - num_imgs_to_show))
            num_imgs_to_show *= 4
            Printer.imshow_with_masks_and_boxes(img_size, *imgs[4 * to_show_id:4 * to_show_id + num_imgs_to_show])
