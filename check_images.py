import torch
from tqdm import tqdm

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
import sys

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image

from train import center_crop_arr

dist.init_process_group("nccl")

# data_path = '/mnt/cephfs/hjh/common_dataset/images/imagenet/ILSVRC2012_img_train'
# data_path = '/mnt/cephfs/hjh/common_dataset/images/imagenet/debug_dataset'
data_path = sys.argv[1]

global_batch_size = 2
num_workers = 96
image_size = 256
rank = dist.get_rank()

read_error_image_files = "/tmp/img_error_files.txt"
open_f = open(read_error_image_files, 'a', encoding='utf-8', buffering=1)


class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        try:
            # This is what ImageFolder normally returns
            original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

            # Get the image path
            path = self.imgs[index][0]

            # Make a new tuple that includes the original and the path
            tuple_with_path = (original_tuple[0], original_tuple[1], path)
            return tuple_with_path

        except (IOError, OSError, Image.DecompressionBombError) as e:
            global open_f
            print(f"Skipping image at index {index}, path {self.imgs[index][0]}: {e}")
            open_f.write(f"{self.imgs[index][0]}\n")
            return None


transform = transforms.Compose([
    transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
])

dataset = ImageFolderWithPaths(data_path, transform=transform)
sampler = DistributedSampler(
    dataset,
    num_replicas=dist.get_world_size(),
    rank=rank,
    shuffle=True,
    seed=12245
)
loader = DataLoader(
    dataset,
    batch_size=int(global_batch_size),
    shuffle=False,
    # sampler=sampler,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True
)

for x, y, p in tqdm(loader):
    # print(f"x:{x}")
    # print(f"y:{y}")
    # print(f"p:{p}")
    # print("-" * 100)
    pass
