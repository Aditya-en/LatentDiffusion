import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class LandscapeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        for category in os.listdir(root_dir):
            category_dir = os.path.join(root_dir, category)
            if os.path.isdir(category_dir):
                for img_name in os.listdir(category_dir):
                    img_path = os.path.join(category_dir, img_name)
                    self.image_paths.append(img_path)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


from torchvision import transforms

image_size = (64, 64)
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Scales to [-1, 1]
])

train_root = './Landscape Classification/Training Data'

train_dataset = LandscapeDataset(root_dir=train_root, transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
)

if __name__ == "__main__":
    print("loading data...")
    batch = next(iter(train_loader))
    print(batch.shape)
