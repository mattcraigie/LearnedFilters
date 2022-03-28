from torch.utils.data import Dataset
import torch


class MultipleBrodatz(Dataset):
    def __init__(self, image_dirs, targets, shuffle=True):
        self.images = []
        for image_dir, target in zip(image_dirs, targets):
            current_image = torch.load(image_dir)
            self.images.append(current_image)

        self.image_data = torch.zeros(size=(len(image_dirs), ) + current_image.shape)
        self.labels = torch.zeros(size=(len(image_dirs), ) + (current_image.shape[0], ), dtype=torch.int64)

        for i in range(len(image_dirs)):
            self.image_data[i] = self.images[i]
            self.labels[i] = torch.full((current_image.shape[0], ), targets[i])
        self.image_data = self.image_data.flatten(0, 1)
        self.labels = self.labels.flatten(0, 1)
        self.image_data = self.image_data.unsqueeze(1)

        if shuffle:
            self.rand_idx = torch.randperm(len(self.labels))
        else:
            self.rand_idx = torch.arange(len(self.labels))

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.image_data[self.rand_idx[idx]], self.labels[self.rand_idx[idx].item()]