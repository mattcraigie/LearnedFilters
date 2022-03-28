from scattering_transform.scattering_transform import ScatteringTransformFast
from scattering_transform.wavelet_models import WaveletsMorlet, CustomFilters
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from datasets import MultipleBrodatz


class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_image_size, output_class_size):
        super(NeuralNetwork, self).__init__()
        self.size = input_image_size
        self.morlet = WaveletsMorlet(input_image_size, J, L)
        self.morlet.make_filter_set()
        self.filters = self.morlet.filter_set_x.flatten(0, 1).unsqueeze(1)
        self.filters = torch.cat([self.filters.real, self.filters.imag], dim=0)
        self.n_filters = self.filters.shape[0]

        # fully connected layer, output 10 classes
        self.fc1 = nn.Linear(4096, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, output_class_size)

    def forward(self, image_batch):
        batch_size = image_batch.shape[0]
        image_batch = nn.functional.pad(image_batch, pad=(self.size, self.size, self.size, self.size), mode='circular')
        x = nn.functional.conv2d(image_batch, self.filters)[:, :, int(self.size/2):int(3 * self.size/2), int(self.size/2):int(3 * self.size/2)]
        x = x.reshape((batch_size, 2, int((self.n_filters // 2) ** 0.5), int((self.n_filters // 2) ** 0.5), self.size, self.size))
        x = torch.norm(x, dim=1)
        x = x.sum(2)
        # x = torch.mean(x, dim=(-2, -1))
        x = x.flatten(-3, -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x.squeeze(1)


class ScatteringNetwork(torch.nn.Module):
    def __init__(self, input_image_size, output_class_size):
        super(ScatteringNetwork, self).__init__()
        self.size = input_image_size
        self.morlet = WaveletsMorlet(input_image_size, J, L)
        self.morlet.make_filter_set()
        self.filters = self.morlet.filter_set_x.flatten(0, 1).unsqueeze(1)
        self.filters = torch.cat([self.filters.real, self.filters.imag], dim=0)
        self.n_filters = self.filters.shape[0]

        self.wavelets = WaveletsMorlet(self.size, 4, 4)
        self.wavelets.make_filter_set()
        self.st = ScatteringTransformFast(self.wavelets)

        # fully connected layer, output 10 classes
        self.fc1 = nn.Linear(11, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, output_class_size)

    def forward(self, image_batch):
        s0, s1, s2 = self.st.run(image_batch)
        s0 = s0[:, None]
        s2 = s2[:, ~s2.isnan()[0]]
        x = torch.cat([s0, s1, s2], dim=1)

        # fig, axes = plt.subplots(ncols=2)
        # axes[0].imshow(image_batch[0])
        # axes[1].plot(x[0])
        # plt.show()


        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x.squeeze(1)


batch_size = 8
size = 32
J = 4
L = 4

train_test_split = 0.8


brodatz_dirs = ['data/brodatz/D9_sliced.pt', 'data/brodatz/D16_sliced.pt']
targets = [0, 1]
data = MultipleBrodatz(brodatz_dirs, targets)

# Creating data indices for training and validation splits:
dataset_size = len(data)
indices = list(range(dataset_size))
split = int(train_test_split * len(data))
train_indices, val_indices = indices[:split], indices[split:]

# Creating PT data samplers and loaders:
train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
test_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=test_sampler)

net = ScatteringNetwork(size, 2)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

losses = []

epochs = 20
training_iteration = 0
for epoch in range(epochs):  # loop over the dataset multiple times
    print(epoch)
    for fields, targets in train_loader:
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(fields.squeeze(1))
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())


print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
    for fields, targets in test_loader:
        outputs = net(fields.squeeze(1))
        predicted = torch.argmax(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

print(outputs)
print(predicted)

print(f'Accuracy of {100 * correct // total} %')

plt.plot(losses)
plt.show()