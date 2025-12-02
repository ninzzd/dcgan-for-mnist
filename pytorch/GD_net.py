# dcgan_mnist_kagglehub.py
"""
DCGAN for MNIST using Kagglehub to fetch dataset (fallback to torchvision).
Save samples to ./samples and checkpoints to ./checkpoints.

Usage:
    python dcgan_mnist_kagglehub.py
"""

import os
import argparse
import math
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils as vutils

# Optional kagglehub import (wrapped so script still runs if not installed)
try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except Exception:
    KAGGLEHUB_AVAILABLE = False

# ---------------------------
# Config / Hyperparameters
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data", help="data directory")
parser.add_argument("--kaggle_slug", type=str, default="pavelg/mnist-in-csv",
                    help="Kaggle dataset slug to fetch (kagglehub). e.g. 'pavelg/mnist-in-csv'")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--image_size", type=int, default=28)
parser.add_argument("--nz", type=int, default=100, help="latent vector size")
parser.add_argument("--ngf", type=int, default=64, help="generator feature maps")
parser.add_argument("--ndf", type=int, default=64, help="discriminator feature maps")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--beta1", type=float, default=0.5)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--out_dir", type=str, default=".")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--use_amp", action="store_true", help="use mixed precision")
args = parser.parse_args()

# Reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

os.makedirs(args.out_dir, exist_ok=True)
SAMPLES_DIR = Path(args.out_dir) / "samples"
CHECKPOINT_DIR = Path(args.out_dir) / "checkpoints"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device(args.device)

# ---------------------------
# Helper: CSV MNIST Dataset (for kaggle CSV)
# ---------------------------
import csv
class MNISTCsvDataset(Dataset):
    """Load MNIST provided as CSV with first column as label and others as pixel values (0-255)."""
    def __init__(self, csv_path: str, transform=None):
        self.csv_path = csv_path
        self.transform = transform
        self._read_index()

    def _read_index(self):
        self.rows = []
        with open(self.csv_path, "r") as f:
            rdr = csv.reader(f)
            header = next(rdr)  # often first row is header
            # If header contains 'label' assume header present
            # We'll treat header as data if it isn't numeric - detect first non-empty line numeric
            # But keep it simple: if header[0].lower() == 'label' skip, else treat as data
            if header and header[0].strip().lower() == "label":
                for r in rdr:
                    if r:
                        self.rows.append(r)
            else:
                # header looked like data, include it
                self.rows.append(header)
                for r in rdr:
                    if r:
                        self.rows.append(r)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        label = int(row[0])
        pixels = np.array(row[1:], dtype=np.uint8).reshape(28, 28)  # MNIST CSV uses 28x28
        img = pixels.astype(np.float32) / 255.0  # [0,1]
        # convert to CxHxW and normalize to [-1,1]
        img = torch.from_numpy(img).unsqueeze(0)  # 1x28x28
        if self.transform:
            img = self.transform(img)
        else:
            img = (img - 0.5) / 0.5
        return img, label

# ---------------------------
# Data loading: try kagglehub then fallback to torchvision
# ---------------------------
transform = transforms.Compose([
    # If we use MNISTCsvDataset, input already float tensor in [0,1], so allow transform to be identity
    transforms.Normalize((0.5,), (0.5,))  # scale to [-1,1]
])

def try_download_kaggle(slug: str, data_dir: str) -> Tuple[str,str]:
    """
    Try to download dataset via kagglehub. Returns (train_csv, test_csv) paths if successful,
    else returns (None, None).
    """
    if not KAGGLEHUB_AVAILABLE:
        print("kagglehub not installed or import failed; falling back to torchvision.")
        return None, None

    print("Attempting to download Kaggle dataset via kagglehub:", slug)
    try:
        # login is interactive if needed, but many environments will have env vars set.
        # We avoid calling kagglehub.login() automatically to prevent blocking scripts.
        resource = kagglehub.resource(slug)
        files = resource.download()  # returns local file paths
        print("Downloaded files from kagglehub:", files)
        # try to detect train/test CSV files
        train_csv = None
        test_csv = None
        for p in files:
            pn = os.path.basename(p).lower()
            if "train" in pn and pn.endswith(".csv"):
                train_csv = p
            if "test" in pn and pn.endswith(".csv"):
                test_csv = p
        # if only one CSV present and name contains 'mnist', assume it's train with header
        if train_csv is None and len(files) == 1 and files[0].lower().endswith(".csv"):
            train_csv = files[0]
        return train_csv, test_csv
    except Exception as e:
        print("kagglehub download failed:", e)
        return None, None

train_csv, test_csv = try_download_kaggle(args.kaggle_slug, args.data_dir)

if train_csv:
    print("Using CSV dataset:", train_csv)
    dataset = MNISTCsvDataset(train_csv, transform=None)  # normalization done in __getitem__
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True)
else:
    print("Using torchvision MNIST (download if needed).")
    mnist = datasets.MNIST(root=args.data_dir, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))
                           ]))
    dataloader = DataLoader(mnist, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True)

# ---------------------------
# DCGAN model (PyTorch)
# ---------------------------
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1):
        super().__init__()
        # DCGAN-style generator for 28x28 output: We'll produce 7x7 intermediate then upsample
        # Architecture: project z -> 128 x 7 x 7 (using ConvTranspose) then upsample to 28x28
        self.net = nn.Sequential(
            # input Z latent vector
            # produce 128 x 7 x 7
            nn.ConvTranspose2d(nz, ngf * 4, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 128 x 7 x 7 -> upsample to 14 x 14
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 64 x 14 x 14 -> upsample to 28 x 28
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # final conv to 1 channel
            nn.Conv2d(ngf, nc, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, nc=1, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            # 1 x 28 x 28
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),  # 14x14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),  # 7x7
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Conv to single output
            nn.Conv2d(ndf * 2, 1, kernel_size=7, stride=1, padding=0, bias=False),  # 1x1
            nn.Sigmoid()
        )

    def forward(self, x):
        # returns batch of scalars
        out = self.net(x)
        return out.view(-1)


# ---------------------------
# Initialize models, optimizers, losses
# ---------------------------
nz = args.nz
ngf = args.ngf
ndf = args.ndf
nc = 1
netG = Generator(nz=nz, ngf=ngf, nc=nc).to(device)
netD = Discriminator(nc=nc, ndf=ndf).to(device)

netG.apply(weights_init)
netD.apply(weights_init)

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

# For AMP
scalerD = torch.cuda.amp.GradScaler(enabled=args.use_amp and device.type == "cuda")
scalerG = torch.cuda.amp.GradScaler(enabled=args.use_amp and device.type == "cuda")

fixed_noise = torch.randn(64, nz, 1, 1, device=device)

real_label_val = 1.0
fake_label_val = 0.0

# ---------------------------
# Training loop
# ---------------------------
print("Starting Training on device:", device)
it = 0
for epoch in range(args.epochs):
    for i, (data, _) in enumerate(dataloader):
        real = data.to(device)
        bs = real.size(0)

        # Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
        netD.zero_grad()
        label = torch.full((bs,), real_label_val, dtype=torch.float, device=device)

        with torch.cuda.amp.autocast(enabled=(args.use_amp and device.type == "cuda")):
            output_real = netD(real)
            lossD_real = criterion(output_real, label)

        if args.use_amp and device.type == "cuda":
            scalerD.scale(lossD_real).backward()
        else:
            lossD_real.backward()

        # Train on fake
        noise = torch.randn(bs, nz, 1, 1, device=device)
        with torch.cuda.amp.autocast(enabled=(args.use_amp and device.type == "cuda")):
            fake = netG(noise)
            label.fill_(fake_label_val)
            output_fake = netD(fake.detach())
            lossD_fake = criterion(output_fake, label)
            lossD = lossD_real + lossD_fake

        if args.use_amp and device.type == "cuda":
            scalerD.scale(lossD_fake).backward()
            scalerD.step(optimizerD)
            scalerD.update()
        else:
            lossD_fake.backward()
            optimizerD.step()

        # Train Generator: maximize log(D(G(z))) <=> minimize BCE with real labels
        netG.zero_grad()
        label.fill_(real_label_val)  # generator wants discriminator to think fakes are real

        with torch.cuda.amp.autocast(enabled=(args.use_amp and device.type == "cuda")):
            output = netD(fake)
            lossG = criterion(output, label)

        if args.use_amp and device.type == "cuda":
            scalerG.scale(lossG).backward()
            scalerG.step(optimizerG)
            scalerG.update()
        else:
            lossG.backward()
            optimizerG.step()

        if i % 200 == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] Batch [{i}/{len(dataloader)}] "
                  f"LossD_real {lossD_real.item():.4f} LossD_fake {lossD_fake.item():.4f} LossG {lossG.item():.4f}")

        # Save samples occasionally
        if it % 500 == 0:
            with torch.no_grad():
                fake_samples = netG(fixed_noise).detach().cpu()
            # denormalize from [-1,1] to [0,1]
            vutils.save_image((fake_samples + 1.0) / 2.0,
                              SAMPLES_DIR / f"sample_iter_{it:06d}.png",
                              nrow=8, normalize=False)
        it += 1

    # Save checkpoint at epoch end
    ckpt = {
        "epoch": epoch + 1,
        "netG_state": netG.state_dict(),
        "netD_state": netD.state_dict(),
        "optimG_state": optimizerG.state_dict(),
        "optimD_state": optimizerD.state_dict()
    }
    torch.save(ckpt, CHECKPOINT_DIR / f"dcgan_epoch_{epoch+1}.pth")
    print("Saved checkpoint:", CHECKPOINT_DIR / f"dcgan_epoch_{epoch+1}.pth")

print("Training finished.")
