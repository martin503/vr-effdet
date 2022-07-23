import torchvision.transforms as T


class Transformator:
    @staticmethod
    def get_transform(img_size, train):
        transform = [
            T.Resize(img_size + img_size // 5),
            T.RandomCrop(img_size)
        ]
        if train:
            transform += [
                T.RandomHorizontalFlip(),
                T.ColorJitter(contrast=0.4, brightness=0.4, saturation=0.25, hue=0.15),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6)),
                T.RandomAutocontrast(),
                T.RandomAdjustSharpness(sharpness_factor=2)
            ]
        transform += [
            T.ToTensor(),
            # T.Normalize(mean=[0.5], std=[0.5]),
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        transform = T.Compose(transform)
        return transform
