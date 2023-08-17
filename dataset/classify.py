from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms


class DatasetClassify(Dataset):
    def __init__(self, data_path, train, download, train_ratio):
        dataset = datasets.MNIST(root=data_path, download=download, train=train)

        if train:
            data_num = int(len(dataset.data.numpy()) * train_ratio)
        else:
            data_num = len(dataset.data.numpy())

        self.images = dataset.data.numpy()[:data_num]
        self.labels = dataset.targets.numpy()[:data_num]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __getitem__(self, index):
        img = self.transform(self.images[index])
        target = self.labels[index]
        return img, target

    def __len__(self):
        return len(self.images)
