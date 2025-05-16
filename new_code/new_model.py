import os
import argparse
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
from pytorch_msssim import SSIM  # pip3 install pytorch-msssim

ssim_loss = SSIM(data_range=1.0, size_average=True, channel=1)

class EndoscopyDepthDataset(Dataset):
    def __init__(self, image_dir, depth_dir, transform=None):
        self.image_paths = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.depth_paths = sorted([
            os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.transform = transform

        assert len(self.image_paths) == len(self.depth_paths), "Mismatch in image and depth count"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        depth = Image.open(self.depth_paths[idx]).convert("L")

        if self.transform:
            image = self.transform(image)
            depth = self.transform(depth)

        depth = depth / (depth.max() + 1e-8)

        return image, depth

# resent 18 + upsampling decoder
class DepthEstimationNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# our combined loss function
def gradient_loss(pred, target):
    def gradient(x):
        D_dx = x[:, :, :, :-1] - x[:, :, :, 1:]
        D_dy = x[:, :, :-1, :] - x[:, :, 1:, :]
        return D_dx, D_dy

    pred_dx, pred_dy = gradient(pred)
    target_dx, target_dy = gradient(target)
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

def combined_loss(pred, target):
    target = F.interpolate(target, size=pred.shape[2:], mode='bilinear', align_corners=False)
    mse = F.mse_loss(pred, target)
    ssim = 1 - ssim_loss(pred, target)
    grad_loss = gradient_loss(pred, target)
    return mse + 0.05 * ssim + 0.2 * grad_loss

# training function
def train(image_dir, depth_dir, epochs, batch_size, lr=1e-4):
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])

    dataset = EndoscopyDepthDataset(image_dir, depth_dir, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthEstimationNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = combined_loss

    print(f"[INFO] Starting training on {len(dataset)} samples")

    for epoch in range(epochs):
        total_loss = 0
        model.train()

        for i, (img, depth) in enumerate(dataloader):
            img, depth = img.to(device), depth.to(device)

            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, depth)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 100 == 0:
                print(f"[DEBUG] Batch {i}, Loss: {loss.item():.6f}, Pred range: {output.min().item():.2f}–{output.max().item():.2f}")

        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "depth_estimation_model.pth")
    print("✅ Training complete. Model saved as 'depth_estimation_model.pth'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a depth estimation model for endoscopy images")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing RGB endoscopy images")
    parser.add_argument("--depth_dir", type=str, required=True, help="Directory containing corresponding depth maps")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    args = parser.parse_args()
    train(args.image_dir, args.depth_dir, args.epochs, args.batch_size, args.lr)