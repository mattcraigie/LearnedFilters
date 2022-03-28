from scattering_transform.scattering_transform import ScatteringTransformFast
from scattering_transform.wavelet_models import WaveletsMorlet, CustomFilters
import torch
import torchvision
import matplotlib.pyplot as plt



class AdaptiveFilters(torch.nn.Module):
    def __init__(self, size, J, L, initial_condition=None, device='cpu'):
        super().__init__()

        self.size = size
        self.J = J
        self.L = L

        start_magnitude = torch.zeros(size=(size, size), requires_grad=True, device=device, dtype=torch.float32)
        start_phase = torch.zeros(size=(size, size), requires_grad=True, device=device, dtype=torch.float32)

        if initial_condition is None:
            morlet = WaveletsMorlet(size, J, L)
            morlet.make_filter_set()
            start_magnitude = start_magnitude + morlet.filter_set_x[0, 0].abs()
            start_phase =  start_phase + morlet.filter_set_x[0, 0].angle()
        else:
            start_magnitude = start_magnitude + initial_condition.abs()
            start_phase = start_phase + initial_condition.angle()

        self.filter_magnitude = torch.nn.Parameter(start_magnitude)
        self.filter_phase = torch.nn.Parameter(start_phase)

    def forward(self):
        return self.filter_magnitude * torch.exp(self.filter_phase * 1j)


class FixedScatteringNetwork(torch.nn.Module):
    def __init__(self, input_image_size, output_class_size, J, L):
        super().__init__()
        self.size = input_image_size
        self.J = J
        self.L = L
        self.fc1 = torch.nn.Linear(11, 4)
        self.fc2 = torch.nn.Linear(4, 10)
        self.filters = WaveletsMorlet(input_image_size, J, L)
        self.filters.make_filter_set()
        self.st = ScatteringTransformFast(self.filters)

    def forward(self, image_batch):
        s0, s1, s2 = self.st.run(image_batch)
        s0 = s0[:, None]
        s2 = s2[:, ~s2.isnan()[0]]
        scattering_coeffs = torch.cat([s0, s1, s2], dim=1)
        x = torch.relu(self.fc1(scattering_coeffs))
        x = torch.sigmoid(self.fc2(x))
        return x



class AdaptiveScatteringNetwork(torch.nn.Module):
    def __init__(self, input_image_size, output_class_size, J, L):
        super().__init__()
        self.size = input_image_size
        self.J = J
        self.L = L
        self.fc1 = torch.nn.Linear(16, 16)
        self.fc2 = torch.nn.Linear(16, output_class_size)
        # self.filter_object = AdaptiveFilters(self.size, J, L)
        self.filters = WaveletsMorlet(input_image_size, J, L)


    def forward(self, image_batch):
        # mother_wavelet = self.filter_object()
        filters = CustomFilters(self.size, self.J, self.L, mother_wavelet=mother_wavelet)
        filters.make_filter_set()
        st = ScatteringTransformFast(filters)
        s0, s1, s2 = self.st.run(image_batch)
        scattering_coeffs = torch.cat([s0, s1, s2.flatten(-1, -2)[~s2.flatten(-1, -2).isnan()]])
        x = torch.relu(self.fc1(scattering_coeffs))
        x = self.fc2(x)
        return x


batch_size = 1024
size = 28
J = 4
L = 4

item_a = 0
item_b = 0

train_test_split = 0.8

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5), (0.5))])


data = torchvision.datasets.MNIST(root='./data', download=True, transform=transform)

criteria = torch.logical_or(data.targets == item_a, data.targets == item_b)
data.data = data.data
data.targets = data.targets

# Creating data indices for training and validation splits:
dataset_size = data.data.shape[0]
indices = list(range(dataset_size))
split = int(train_test_split * data.data.shape[0])
train_indices, val_indices = indices[:split], indices[split:]

# Creating PT data samplers and loaders:
train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
test_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=test_sampler)

net = FixedScatteringNetwork(size, 10, J, L)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-20)

losses = []

epochs = 1
training_iteration = 0
for epoch in range(epochs):  # loop over the dataset multiple times
    for fields, targets in train_loader:

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(fields.squeeze(dim=1))

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
    for fields, targets in test_loader:
        # calculate outputs by running images through the network
        outputs = net(fields.squeeze(1))

        # the class with the highest energy is what we choose as prediction
        predicted = torch.argmax(outputs.data, 1)

        total += targets.size(0)
        correct += (predicted == targets).sum().item()

print(f'Accuracy of {100 * correct // total} %')

plt.plot(losses)
plt.show()