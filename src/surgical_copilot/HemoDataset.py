import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np

class HemoDataset(Dataset):
    def __init__(self, root_dir="data/raw", transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        self.image_paths = sorted(list(self.root_dir.rglob("*/imgs/*.png")))
        self.mask_paths = sorted(list(self.root_dir.rglob("*/labels/*.png")))

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"Nessuna immagine trovata in {root_dir}")
            
        if len(self.image_paths) != len(self.mask_paths):
            raise RuntimeError(
                f"Mismatch: {len(self.image_paths)} immagini vs {len(self.mask_paths)} maschere."
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Caricamento e conversione immediata in numpy per compatibilità MONAI
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        
        # Aggiunta dimensione canale per MONAI (H, W) -> (C, H, W)
        image = np.transpose(image, (2, 0, 1)) # Da HWC a CHW
        mask = np.expand_dims(mask, axis=0)    # Da HW a 1HW

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return torch.as_tensor(image).float(), torch.as_tensor(mask).float()