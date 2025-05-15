import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


# --- Model (same as training) ---
class DepthEstimationNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, 2, stride=2), torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, 2, stride=2), torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 1, 2, stride=2)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# --- Dataset ---
class EndoscopyDepthDataset(Dataset):
    def __init__(self, image_dir, depth_dir, transform=None):
        self.image_paths = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))
        ])
        self.depth_paths = sorted([
            os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith(('.png', '.jpg'))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        depth = Image.open(self.depth_paths[idx]).convert("L")

        if self.transform:
            image = self.transform(image)
            depth = self.transform(depth)
        
        # Normalize depth to [0, 1]
        depth = depth / depth.max()

        return image, depth


# --- Evaluation Metrics ---
def compute_metrics(pred, target):
    pred = pred.squeeze().cpu().numpy()
    target = target.squeeze().cpu().numpy()

    mse = np.mean((pred - target) ** 2)
    mae = np.mean(np.abs(pred - target))
    rmse = np.sqrt(mse)
    ssim_val = ssim(pred, target, data_range=target.max() - target.min())

    return mse, mae, rmse, ssim_val


# --- Heatmap Saving ---
def save_heatmap(pred, target, idx, output_dir):
    pred = pred.squeeze().cpu().numpy()
    target = target.squeeze().cpu().numpy()
    
    # Compute absolute error map
    abs_error = np.abs(pred - target)
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    axs[0].imshow(pred, cmap='inferno')
    axs[0].set_title("Predicted Depth")
    axs[0].axis('off')

    axs[1].imshow(target, cmap='inferno')
    axs[1].set_title("Ground Truth Depth")
    axs[1].axis('off')

    axs[2].imshow(abs_error, cmap='viridis')
    axs[2].set_title("Absolute Error Map")
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"heatmap_{idx:03d}.png"))
    plt.close()


# --- Main ---
def evaluate(model_path, image_dir, depth_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = EndoscopyDepthDataset(image_dir, depth_dir, transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthEstimationNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    all_metrics = []

    for idx, (img, gt_depth) in enumerate(dataloader):
        img = img.to(device)
        gt_depth = gt_depth.to(device)

        with torch.no_grad():
            pred_depth = model(img)
        
        pred_depth = pred_depth * gt_depth.max()
        pred_depth_np = pred_depth.squeeze().cpu().numpy()

        raw_depth_output_path = os.path.join(output_dir, 'output_raw', f"raw_depth_{idx:03d}.npy")

        np.save(raw_depth_output_path, pred_depth_np)

        mse, mae, rmse, ssim_val = compute_metrics(pred_depth, gt_depth)
        all_metrics.append((mse, mae, rmse, ssim_val))

        save_heatmap(pred_depth, gt_depth, idx, output_dir)

    # Print averaged metrics
    if len(all_metrics) == 0:
        print("No images were processed. Please check your image/depth directories.")
        return

    all_metrics = np.array(all_metrics)
    print(f"Average MSE:   {all_metrics[:, 0].mean():.4f}")
    print(f"Average MAE:   {all_metrics[:, 1].mean():.4f}")
    print(f"Average RMSE:  {all_metrics[:, 2].mean():.4f}")
    print(f"Average SSIM:  {all_metrics[:, 3].mean():.4f}")


# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--depth_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./heatmaps")

    args = parser.parse_args()
    evaluate(args.model_path, args.image_dir, args.depth_dir, args.output_dir)
