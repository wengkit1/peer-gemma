"""
Real data loading module for PEER Gemma training on NSCC
Uses HuggingFace datasets - NO tokenization here, model handles it
"""
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Optional
from loguru import logger
import os


class TokenDataset(Dataset):
    """Dataset that loads raw text - model will tokenize during forward pass"""

    def __init__(
            self,
            dataset_name: str = "wikitext",
            dataset_config: str = "wikitext-2-raw-v1",
            split: str = "train",
            sequence_length: int = 256,
            max_samples: Optional[int] = None,
            cache_dir: Optional[str] = None
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.sequence_length = sequence_length
        self.max_samples = max_samples
        self.cache_dir = cache_dir or os.getenv("HF_DATASETS_CACHE", "~/scratch")

        # Load raw text data
        self._load_raw_data()

    def _load_raw_data(self):
        """Load raw text without tokenization"""
        logger.info(f"Loading dataset: {self.dataset_name}/{self.dataset_config}, split: {self.split}")

        try:
            # Load dataset
            if self.dataset_config:
                dataset = load_dataset(
                    self.dataset_name,
                    self.dataset_config,
                    split=self.split,
                    cache_dir=self.cache_dir,
                    token=os.getenv("HF_TOKEN")
                )
            else:
                dataset = load_dataset(
                    self.dataset_name,
                    split=self.split,
                    cache_dir=self.cache_dir,
                    token=os.getenv("HF_TOKEN")
                )

            logger.info(f"Dataset loaded: {len(dataset)} samples")

            # Limit samples if specified
            if self.max_samples and len(dataset) > self.max_samples:
                dataset = dataset.select(range(self.max_samples))
                logger.info(f"Limited to {self.max_samples} samples")

            # Extract raw text - NO tokenization/mapping
            self.texts = []
            for item in dataset:
                # Handle different dataset formats
                text = None
                if 'text' in item:
                    text = item['text']
                elif 'content' in item:
                    text = item['content']
                else:
                    # Try to find the text column
                    text_columns = [col for col in item.keys() if 'text' in col.lower()]
                    if text_columns:
                        text = item[text_columns[0]]

                # Filter out empty or very short texts
                if text and isinstance(text, str) and len(text.strip()) > 10:
                    self.texts.append(text.strip())

            logger.info(f"Extracted {len(self.texts)} text samples")

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        """Return raw text string"""
        return self.texts[idx]


class DataModule(pl.LightningDataModule):
    """Lightning data module for real datasets"""

    def __init__(
            self,
            dataset_name: str = "wikitext",
            dataset_config: str = "wikitext-2-raw-v1",
            sequence_length: int = 256,
            vocab_size: int = None,  # Not used, kept for compatibility
            batch_size: int = 4,
            num_samples: Optional[int] = None,
            num_workers: int = 2,
            pin_memory: bool = True,
            persistent_workers: bool = True,
            cache_dir: Optional[str] = None,
            seed: int = 42,
            **kwargs  # Accept additional kwargs for compatibility
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.seed = seed
        self.cache_dir = cache_dir or os.getenv("HF_DATASETS_CACHE", "~/scratch")

        # Platform-specific optimizations
        import torch
        import platform

        if torch.backends.mps.is_available() and pin_memory:
            logger.info("Detected MPS (Mac), disabling pin_memory to avoid warnings")
            self.pin_memory = False
        else:
            self.pin_memory = pin_memory

        self.num_workers = num_workers
        self.persistent_workers = persistent_workers and num_workers > 0

    def setup(self, stage: str = None):
        """Setup datasets for different stages"""

        if stage == "fit" or stage is None:
            # Train dataset
            self.train_dataset = TokenDataset(
                dataset_name=self.dataset_name,
                dataset_config=self.dataset_config,
                split="train",
                sequence_length=self.sequence_length,
                max_samples=int(0.8 * self.num_samples) if self.num_samples else None,
                cache_dir=self.cache_dir
            )

            # Validation dataset
            try:
                self.val_dataset = TokenDataset(
                    dataset_name=self.dataset_name,
                    dataset_config=self.dataset_config,
                    split="validation",
                    sequence_length=self.sequence_length,
                    max_samples=int(0.1 * self.num_samples) if self.num_samples else 100,
                    cache_dir=self.cache_dir
                )
            except:
                # If no validation split, use a portion of train
                logger.warning("No validation split found, using portion of train data")
                self.val_dataset = TokenDataset(
                    dataset_name=self.dataset_name,
                    dataset_config=self.dataset_config,
                    split="train[:10%]",
                    sequence_length=self.sequence_length,
                    max_samples=100,
                    cache_dir=self.cache_dir
                )

        if stage == "test" or stage is None:
            try:
                self.test_dataset = TokenDataset(
                    dataset_name=self.dataset_name,
                    dataset_config=self.dataset_config,
                    split="test",
                    sequence_length=self.sequence_length,
                    max_samples=int(0.1 * self.num_samples) if self.num_samples else 100,
                    cache_dir=self.cache_dir
                )
            except:
                # Use validation as test if no test split
                self.test_dataset = self.val_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False
        )


def create_data_module(**kwargs) -> DataModule:
    """Create a DataModule with platform-optimized settings"""
    import torch
    import platform

    # Default settings for real data
    defaults = {
        'dataset_name': 'wikitext',
        'dataset_config': 'wikitext-2-raw-v1',
        'sequence_length': 256,
        'batch_size': 4,
        'num_samples': 5000,
        'num_workers': 2,
        'pin_memory': True,
        'persistent_workers': True,
        'seed': 42
    }

    # Override defaults with provided kwargs
    defaults.update(kwargs)

    # Platform-specific optimizations
    if torch.backends.mps.is_available():
        logger.info("🍎 Detected Mac with MPS - optimizing settings")
        defaults['pin_memory'] = False
        defaults['num_workers'] = min(2, defaults['num_workers'])
    elif platform.system() == "Darwin":  # Mac without MPS
        logger.info("🍎 Detected Mac (CPU only) - optimizing settings")
        defaults['pin_memory'] = False
        defaults['num_workers'] = 0
        defaults['persistent_workers'] = False
    elif torch.cuda.is_available():
        logger.info("🚀 Detected CUDA - using optimized GPU settings")
        defaults['pin_memory'] = True
    else:
        logger.info("💻 Detected CPU-only - optimizing settings")
        defaults['pin_memory'] = False
        defaults['num_workers'] = min(2, defaults['num_workers'])

    return DataModule(**defaults)


def test_data():
    """Test function for real data loading"""
    logger.info("Testing real data loading...")

    try:
        # Test with small dataset
        data_module = create_data_module(
            dataset_name='allenai/c4',
            dataset_config='en',
            sequence_length=128,
            batch_size=2,
            num_samples=100,
            num_workers=1
        )

        logger.info("Setting up data module...")
        data_module.setup("fit")

        logger.info("Testing train dataloader...")
        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))

        logger.info(f"Batch type: {type(batch)}")
        logger.info(f"Batch length: {len(batch)}")
        logger.info(f"Sample text preview: {batch[0][:100] if batch[0] else 'None'}...")

        # Validate batch contains text strings
        assert isinstance(batch, (list, tuple)), f"Expected list/tuple, got {type(batch)}"
        assert len(batch) == 2, f"Expected batch size 2, got {len(batch)}"
        assert all(isinstance(text, str) for text in batch), "Expected all items to be strings"

        logger.success("✅ Real data test passed!")
        return True

    except Exception as e:
        logger.error(f"❌ Real data test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    test_data()