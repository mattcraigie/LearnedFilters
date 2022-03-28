from scattering_transform.scattering_transform import ScatteringTransformFast
from scattering_transform.wavelet_models import WaveletsMorlet
import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from datasets import MultipleBrodatz
import time

class AdaptiveFilters(torch.nn.Module):
    def __init__(self, size, J, L, initial_condition=None, device='cpu'):
        super().__init__()

        self.size = size
        self.J = J
        self.L = L

        if initial_condition is None:
            initial_condition = WaveletsMorlet(size, J, L).filters_cut

        self.filter_magnitudes = nn.ParameterList([])
        self.filter_phases = nn.ParameterList([])

        for j in range(J):
            self.filter_magnitudes.append(torch.nn.Parameter(initial_condition[j].abs(), requires_grad=True))
            self.filter_phases.append(torch.nn.Parameter(initial_condition[j].angle(), requires_grad=True))

    def forward(self):
        self.filters_cut = [mags * torch.exp(phases * 1j) for mags, phases in zip(self.filter_magnitudes, self.filter_phases)]
        return


class AdaptiveScatteringNetwork(torch.nn.Module):
    def __init__(self, input_image_size, output_class_size, J, L):
        super(AdaptiveScatteringNetwork, self).__init__()
        self.size = input_image_size
        self.J = J
        self.L = L
        self.adaptive_filters = AdaptiveFilters(self.size, J, L)

        # fully connected layer
        self.fc1 = nn.Linear(22, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, output_class_size)

    def forward(self, image_batch):
        self.adaptive_filters()
        st = ScatteringTransformFast(self.adaptive_filters)

        s0, s1, s2 = st.run(image_batch)
        s0 = s0[:, None]
        s2 = s2[:, ~s2.isnan()[0]]
        x = torch.cat([s0, s1, s2], dim=1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x.squeeze(1)

class StandardScatteringNetwork(torch.nn.Module):
    def __init__(self, input_image_size, output_class_size, J, L):
        super(StandardScatteringNetwork, self).__init__()
        self.size = input_image_size
        self.J = J
        self.L = L
        self.st = ScatteringTransformFast(WaveletsMorlet(size, J, L))

        # fully connected layer
        self.fc1 = nn.Linear(22, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, output_class_size)

    def forward(self, image_batch):
        s0, s1, s2 = self.st.run(image_batch)
        s0 = s0[:, None]
        s2 = s2[:, ~s2.isnan()[0]]
        x = torch.cat([s0, s1, s2], dim=1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x.squeeze(1)

torch.manual_seed(1)
# nice_combo: rs 1, first three, 3 epochs, lr=1e-3


epochs = 10
batch_size = 16
lr = 1e-2

size = 128
J = 6
L = 4

train_test_split = 0.8

n_brodatz = 2
brodatz_dirs = ['data/brodatz/D{}_sliced.pt'.format(i) for i in range(1, n_brodatz + 1)]
targets = list(range(n_brodatz))
data = MultipleBrodatz(brodatz_dirs, targets)

def show_data(a):
    fig, axes = plt.subplots(nrows=2, ncols=3)

    for i in range(2):
        for j in range(3):
            axes[i, j].imshow(data.image_data[a[i, j], 0], cmap='gray')
            axes[i, j].set_axis_off()
    plt.show()

show_data(np.array([[11, 111, 211], [-1, -271, -311]]))


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

# net = AdaptiveScatteringNetwork(size, len(targets), J, L)
net = StandardScatteringNetwork(size, len(targets), J, L)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(net.parameters(), lr=lr)

losses = []

print("Starting Training")
start = time.time()
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

end = time.time()
print('Finished Training')
print('Time elapsed: ', end - start)

correct = 0
total = 0
with torch.no_grad():
    for fields, targets in test_loader:
        outputs = net(fields.squeeze(1))
        predicted = torch.argmax(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

print(f'Accuracy of {100 * correct // total} %')

plt.plot(losses)
plt.title("Adaptive Filter Network Training Loss")
plt.xlabel("Training batch number")
plt.ylabel("Training loss")
plt.show()

filters = net.adaptive_filters.filters_cut

# for i in range(4):
#     for j in range(4):
#         fig, axes = plt.subplots(ncols=2, nrows=2)
#         values = torch.fft.fftshift(filters[i][j].detach())
#
#         phase = values.angle()
#         phase[phase > torch.pi/2] -= torch.pi
#         phase[phase < -torch.pi/2] += torch.pi
#
#         axes[0, 0].imshow(values.abs())
#         axes[1, 0].imshow(phase)
#         axes[0, 1].imshow(torch.fft.fftshift(torch.fft.ifft2(filters[i][j].detach()).real))
#         axes[1, 1].imshow(torch.fft.fftshift(torch.fft.ifft2(filters[i][j].detach()).imag))
#
#         for a in range(2):
#             for b in range(2):
#                 axes[a, b].set_axis_off()
#
#         plt.show()

idx = [111, -271]
c = ['blue', 'green']

fig, axes = plt.subplots(nrows=2, sharex=True)

for n, i in enumerate(idx):
    st = ScatteringTransformFast(net.adaptive_filters)
    s0, s1, s2 = st.run(data.image_data[i])
    s0 = s0[:, None]
    s2 = s2[:, ~s2.isnan()[0]]
    x = torch.cat([s0, s1, s2], dim=1)
    axes[1].plot(torch.log(x[0]), c=c[n])

    st = ScatteringTransformFast(WaveletsMorlet(size, J, L))
    s0, s1, s2 = st.run(data.image_data[i])
    s0 = s0[:, None]
    s2 = s2[:, ~s2.isnan()[0]]
    x = torch.cat([s0, s1, s2], dim=1)
    axes[0].plot(torch.log(x[0]), c=c[n])

    axes[1].set_title("After")
    axes[1].set_xlabel("Scales for 0th, 1st, 2nd order ($j^0$, $j^1_x$, $j^2_{xx}$)")
    axes[1].set_ylabel("log ($s^0$, $s^1$, $s^2$)")
    axes[0].set_ylabel("log ($s^0$, $s^1$, $s^2$)")
    axes[0].set_title("Before")

plt.suptitle("Scattering Coefficients before and after adaptive filter training")
plt.show()
