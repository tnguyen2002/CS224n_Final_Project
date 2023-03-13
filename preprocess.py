"""
Given a set of folders and patches, convert
patches into embeddings from ResNet-50 model 
and save out as .pt files.

TODO: Modify slide, to save slides directly out as .pt
files, upload patches to S3? for analysis? How to get maps
for attention analysis
"""
# %%
import argparse
import torch
import torchvision.transforms as transforms
from pathlib import Path
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from tqdm import tqdm


class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.image_paths = list(self.root_dir.glob('*.jpg')) + list(self.root_dir.glob('*.jpeg')) + list(self.root_dir.glob('*.png'))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

parser = argparse.ArgumentParser(description='Convert directory of image patches to ResNet-50 features')
parser.add_argument('--input-dir', type=str, help='Directory containing folders of image patches')
parser.add_argument('--output-dir', type=str, help='Directory to save ResNet-50 features as .pt files')
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
print("Model: ", model)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Model is on ", device)
model.eval().to(device)

# %%

for folder_path in Path(args.input_dir).iterdir():
    output_path = output_dir / f'{folder_path.stem}.pt'
    if not folder_path.is_dir() or output_path.exists():
        continue

    print(f'Processing folder: {folder_path.name}')
    dataset = ImageFolderDataset(folder_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    features = []
    with torch.no_grad():
        for images in tqdm(dataloader):
            images = images.to(device)
            output = model(images)
            features.append(output)
    if len(features) == 0: 
        print("No features... skipping.")
        continue
    
    features = torch.cat(features, dim=0)
    print("Features shape: ", features.shape)

    
    torch.save(features, output_path)
    print(f'Saved features for {folder_path.stem} to {output_path}')

