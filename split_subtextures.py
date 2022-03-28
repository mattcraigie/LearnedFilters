import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToTensor
from PIL import Image
from scipy.ndimage import rotate

def load_tif(file):
    im1 = Image.open(file)
    return ToTensor()(im1).squeeze(0)

def slice_image(image, n_slices):
    im_size, _ = image.shape
    slice_size = im_size // n_slices

    result = torch.zeros((n_slices, n_slices, slice_size, slice_size))

    for i in range(n_slices):
        for j in range(n_slices):
            i_staggered = i + torch.rand((1,)) - 0.5  # stagger so they are semi-randomly sampled but roughly cover the whole image
            j_staggered = j + torch.rand((1,)) - 0.5

            # shift the whole image around to avoid dealing with wrapping
            rolled = image.roll((-int(slice_size * i_staggered), -int(slice_size * j_staggered)), dims=(-2, -1))
            result[i, j] = rolled[:slice_size, :slice_size]

    return result.flatten(0, 1)


def make_slices(infile, outfile, n_slices, n_rotations):
    image = load_tif(infile)
    slice_size = image.shape[-1] // n_slices
    result = torch.zeros((n_rotations,n_slices**2, slice_size, slice_size))
    angles = torch.rand(n_rotations) * 360  # angle is in degrees

    for i in range(n_rotations):
        image = image.roll((torch.randint(image.shape[-2], (1,)).item(),
                            torch.randint(image.shape[-1], (1,)).item()), dims=(-2, -1))  # random translation
        rotated = rotate(image, angle=angles[i].item(), mode='wrap', reshape=False)  # random rotation
        rotated = torch.tensor(rotated)
        sliced = slice_image(rotated, n_slices)
        result[i] = sliced
    result = result.flatten(0, 1)
    torch.save(result, outfile)


infiles = ['data/brodatz/D{}.tif'.format(i) for i in range(1, 10)]
outfiles = ['data/brodatz/D{}_sliced.pt'.format(i) for i in range(1, 10)]

n_slices, n_rotations = 5, 20
for infile, outfile in zip(infiles, outfiles):
    print(infile)
    make_slices(infile, outfile, n_slices, n_rotations)