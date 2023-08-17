import random

from PIL import Image
from PIL import ImageFilter
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms


class GaussianBlur(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class DatasetPretrain(Dataset):
    def __init__(self, data_path, train, download):
        dataset = datasets.MNIST(root=data_path, download=download, train=train)
        self.images = dataset.data.numpy()
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(28, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __getitem__(self, index):
        img = Image.fromarray(self.images[index])
        img1 = self.transform(img)
        img2 = self.transform(img)
        return img1, img2

    def __len__(self):
        return len(self.images)
