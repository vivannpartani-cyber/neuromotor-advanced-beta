import h5py
import torch
from torch.utils.data import Dataset
import os

class NSDDataset(Dataset):
    """
    High-performance PyTorch Dataset for the Natural Scenes Dataset (NSD).
    Lazy-loads massive HDF5 files to prevent RAM exhaustion during cluster training.
    """
    def __init__(self, fmri_path: str, clip_path: str, subject: str = "subj01", split: str = "train"):
        """
        Args:
            fmri_path: Path to the HDF5 file containing fMRI betas.
            clip_path: Path to the HDF5 file containing CLIP image embeddings.
            subject: The NSD subject ID (e.g., 'subj01').
            split: 'train' or 'test'.
        """
        super().__init__()
        self.fmri_path = fmri_path
        self.clip_path = clip_path
        self.subject = subject
        self.split = split
        
        # Verify files exist before training starts
        if not os.path.exists(fmri_path):
            raise FileNotFoundError(f"fMRI data not found at {fmri_path}")
        if not os.path.exists(clip_path):
            raise FileNotFoundError(f"CLIP data not found at {clip_path}")

        # Open files to get dataset length
        with h5py.File(self.fmri_path, 'r') as f_fmri:
            self.num_samples = f_fmri[self.split].shape[0]

        # The actual HDF5 handles will be opened per-worker in __getitem__ to be multiprocessing-safe
        self.fmri_file = None
        self.clip_file = None

    def _init_files(self):
        """Open HDF5 handles lazily per worker."""
        if self.fmri_file is None:
            self.fmri_file = h5py.File(self.fmri_path, 'r')
            self.clip_file = h5py.File(self.clip_path, 'r')

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        self._init_files()
        
        # Load exactly one sample from disk
        fmri = self.fmri_file[self.split][idx]
        clip_emb = self.clip_file[self.split][idx]
        
        # Convert to PyTorch tensors
        fmri_tensor = torch.tensor(fmri, dtype=torch.float32)
        clip_tensor = torch.tensor(clip_emb, dtype=torch.float32)
        
        return fmri_tensor, clip_tensor

    def close(self):
        if self.fmri_file is not None:
            self.fmri_file.close()
        if self.clip_file is not None:
            self.clip_file.close()
