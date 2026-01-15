import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Define transforms with your specific center crop
transform = transforms.Compose([
    transforms.Lambda(lambda img: transforms.functional.crop(img, top=39, left=9, height=160, width=160)),
    transforms.Resize(128),  # Resize to 32x32 to match your model
    transforms.ToTensor(),  # Convert to [0, 1] range
])
    
# Create dataset and dataloader
celeba_root = '/ssd_scratch/cvit/souvik/CelebA/CelebA/Img/img_align_celeba'


class CelebADataset(Dataset):
    def __init__(self, root_dir, split_file, split="train", transform=None):
        """
        Args:
            root_dir (str): Path to img_align_celeba folder
            split_file (str): Path to CelebA split file (e.g., list_eval_partition.txt)
            split (str): One of ['train', 'val', 'test']
            transform: Optional torchvision transforms
        """
        assert split in ["train", "val", "test"]

        self.root_dir = root_dir
        self.transform = transform

        split_map = {
            "train": 0,
            "val": 1,
            "test": 2
        }
        target_split = split_map[split]

        self.image_files = []

        with open(split_file, "r") as f:
            for line in f:
                fname, split_id = line.strip().split()
                if int(split_id) == target_split:
                    self.image_files.append(fname)

        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Dummy label to stay CIFAR-like
        return image, 0


# dataset = CelebADataset(
#     root_dir=celeba_root,
#     transform=transform
# )

# data_loader = DataLoader(
#     dataset,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=4,
#     pin_memory=True
# )

# print(f"Total CelebA images: {len(dataset)}")

# # Test loading one batch
# test_batch, _ = next(iter(data_loader))
# print(f"Batch shape: {test_batch.shape}")  # Should be [batch_size, 3, 32, 32]