# File: data/mock_data.py
"""
Mock data generation for testing PEER Gemma training
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import List, Tuple
from loguru import logger


class MockTokenDataset(Dataset):
    """Mock dataset that generates various token patterns"""

    def __init__(
            self,
            num_samples: int,
            sequence_length: int,
            vocab_size: int,
            patterns: List[str] = None,
            seed: int = 42
    ):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.seed = seed

        if patterns is None:
            patterns = ["repeat", "arithmetic", "random", "structured"]
        self.patterns = patterns

        # Set random seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)

        logger.info(f"Created MockTokenDataset: {num_samples} samples, "
                    f"seq_len={sequence_length}, vocab={vocab_size}")
        logger.info(f"Patterns: {patterns}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a mock sequence based on the pattern"""

        # Choose pattern based on index
        pattern = self.patterns[idx % len(self.patterns)]

        if pattern == "repeat":
            sequence = self._generate_repeat_pattern(idx)
        elif pattern == "arithmetic":
            sequence = self._generate_arithmetic_pattern(idx)
        elif pattern == "structured":
            sequence = self._generate_structured_pattern(idx)
        else:  # random
            sequence = self._generate_random_pattern(idx)

        # Create input and target (shifted by 1 for language modeling)
        input_ids = sequence[:-1]
        target_ids = sequence[1:]

        return input_ids, target_ids

    def _generate_repeat_pattern(self, idx: int) -> torch.Tensor:
        """Generate repeating sequences like [1,2,3,1,2,3,...]"""
        # Fix: ensure pattern length is reasonable
        pattern_length = min(10, max(2, self.vocab_size // 10))
        base_pattern = torch.randint(0, self.vocab_size, (pattern_length,))

        # Repeat pattern to fill sequence
        repeats = (self.sequence_length // pattern_length) + 1
        sequence = base_pattern.repeat(repeats)[:self.sequence_length]

        return sequence

    def _generate_arithmetic_pattern(self, idx: int) -> torch.Tensor:
        """Generate arithmetic sequences like [1,2,3,4,5,...]"""
        start = torch.randint(0, self.vocab_size // 2, (1,)).item()

        # Fix: ensure step range is always valid
        max_step = max(2, min(5, self.vocab_size // 20))  # At least 2, reasonable upper bound
        step = torch.randint(1, max_step, (1,)).item()

        sequence = torch.arange(
            start,
            start + step * self.sequence_length,
            step
        ) % self.vocab_size

        return sequence[:self.sequence_length]

    def _generate_structured_pattern(self, idx: int) -> torch.Tensor:
        """Generate structured patterns with some logic"""
        sequence = torch.zeros(self.sequence_length, dtype=torch.long)

        # Create a simple pattern: even positions get even numbers, odd get odd
        # Fix: handle small vocab sizes properly
        half_vocab = max(1, self.vocab_size // 2)

        for i in range(self.sequence_length):
            if i % 2 == 0:
                # Even positions: even numbers
                even_num = torch.randint(0, half_vocab, (1,)) * 2
                sequence[i] = even_num % self.vocab_size
            else:
                # Odd positions: odd numbers
                odd_num = torch.randint(0, half_vocab, (1,)) * 2 + 1
                sequence[i] = odd_num % self.vocab_size

        return sequence

    def _generate_random_pattern(self, idx: int) -> torch.Tensor:
        """Generate completely random sequences"""
        return torch.randint(0, self.vocab_size, (self.sequence_length,))


class MockDataModule(pl.LightningDataModule):
    """Lightning data module for mock token data"""

    def __init__(
            self,
            num_samples: int = 1000,
            sequence_length: int = 256,
            vocab_size: int = 2000,
            batch_size: int = 4,
            patterns: List[str] = None,
            seed: int = 42,
            num_workers: int = 2,
            pin_memory: bool = True,
            persistent_workers: bool = True
    ):
        super().__init__()
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.patterns = patterns
        self.seed = seed
        self.num_workers = num_workers

        # Smart pin_memory: disable on MPS (Mac) to avoid warnings
        import torch
        if torch.backends.mps.is_available() and pin_memory:
            logger.info("Detected MPS (Mac), disabling pin_memory to avoid warnings")
            self.pin_memory = False
        else:
            self.pin_memory = pin_memory

        # Smart persistent_workers: only if num_workers > 0
        self.persistent_workers = persistent_workers and num_workers > 0

        # Split data
        self.train_samples = int(0.8 * num_samples)
        self.val_samples = int(0.1 * num_samples)
        self.test_samples = num_samples - self.train_samples - self.val_samples

        logger.info(f"MockDataModule: train={self.train_samples}, "
                    f"val={self.val_samples}, test={self.test_samples}")
        logger.info(f"DataLoader settings: pin_memory={self.pin_memory}, "
                    f"num_workers={self.num_workers}, persistent_workers={self.persistent_workers}")

    def setup(self, stage: str = None):
        """Setup datasets for different stages"""

        if stage == "fit" or stage is None:
            self.train_dataset = MockTokenDataset(
                self.train_samples,
                self.sequence_length,
                self.vocab_size,
                self.patterns,
                self.seed
            )

            self.val_dataset = MockTokenDataset(
                self.val_samples,
                self.sequence_length,
                self.vocab_size,
                self.patterns,
                self.seed + 1  # Different seed for validation
            )

        if stage == "test" or stage is None:
            self.test_dataset = MockTokenDataset(
                self.test_samples,
                self.sequence_length,
                self.vocab_size,
                self.patterns,
                self.seed + 2  # Different seed for test
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0
        )


def create_smart_data_module(**kwargs) -> MockDataModule:
    """Create a MockDataModule with platform-optimized settings"""
    import torch
    import platform

    # Default settings
    defaults = {
        'num_workers': 2,
        'pin_memory': True,
        'persistent_workers': True
    }

    # Override defaults with provided kwargs
    defaults.update(kwargs)

    # Platform-specific optimizations
    if torch.backends.mps.is_available():
        logger.info("🍎 Detected Mac with MPS - optimizing settings")
        defaults['pin_memory'] = False  # MPS doesn't support pin_memory
        defaults['num_workers'] = min(2, defaults['num_workers'])  # Conservative on Mac
    elif platform.system() == "Darwin":  # Mac without MPS
        logger.info("🍎 Detected Mac (CPU only) - optimizing settings")
        defaults['pin_memory'] = False
        defaults['num_workers'] = 0  # Avoid multiprocessing issues on Mac
        defaults['persistent_workers'] = False
    elif torch.cuda.is_available():
        logger.info("🚀 Detected CUDA - using optimized GPU settings")
        defaults['pin_memory'] = True
        # Keep other defaults
    else:
        logger.info("💻 Detected CPU-only - optimizing settings")
        defaults['pin_memory'] = False
        defaults['num_workers'] = min(2, defaults['num_workers'])

    return MockDataModule(**defaults)


def create_smart_data_module(**kwargs) -> MockDataModule:
    """Create a MockDataModule with platform-optimized settings"""
    import torch
    import platform

    # Default settings
    defaults = {
        'num_workers': 2,
        'pin_memory': True,
        'persistent_workers': True
    }

    # Override defaults with provided kwargs
    defaults.update(kwargs)

    # Platform-specific optimizations
    if torch.backends.mps.is_available():
        logger.info("🍎 Detected Mac with MPS - optimizing settings")
        defaults['pin_memory'] = False  # MPS doesn't support pin_memory
        defaults['num_workers'] = min(2, defaults['num_workers'])  # Conservative on Mac
    elif platform.system() == "Darwin":  # Mac without MPS
        logger.info("🍎 Detected Mac (CPU only) - optimizing settings")
        defaults['pin_memory'] = False
        defaults['num_workers'] = 0  # Avoid multiprocessing issues on Mac
        defaults['persistent_workers'] = False
    elif torch.cuda.is_available():
        logger.info("🚀 Detected CUDA - using optimized GPU settings")
        defaults['pin_memory'] = True
        # Keep other defaults
    else:
        logger.info("💻 Detected CPU-only - optimizing settings")
        defaults['pin_memory'] = False
        defaults['num_workers'] = min(2, defaults['num_workers'])

    return MockDataModule(**defaults)


def test_mock_data():
    """Test function for mock data generation"""
    logger.info("Testing mock data generation...")

    # Test with different vocab sizes to ensure robustness
    test_configs = [
        {"vocab_size": 50, "sequence_length": 20, "name": "Small vocab"},
        {"vocab_size": 100, "sequence_length": 32, "name": "Medium vocab"},
        {"vocab_size": 1000, "sequence_length": 64, "name": "Large vocab"},
    ]

    for config in test_configs:
        logger.info(f"Testing {config['name']} (vocab={config['vocab_size']})...")

        # Create test dataset
        dataset = MockTokenDataset(
            num_samples=10,
            sequence_length=config['sequence_length'],
            vocab_size=config['vocab_size'],
            patterns=["repeat", "arithmetic", "structured", "random"]
        )

        # Test a few samples
        for i in range(4):  # Test each pattern type
            try:
                input_ids, target_ids = dataset[i]
                pattern_name = dataset.patterns[i % len(dataset.patterns)]

                logger.info(f"  {pattern_name}: input shape {input_ids.shape}, target shape {target_ids.shape}")

                # Validate shapes
                expected_len = config['sequence_length'] - 1
                assert input_ids.shape == (
                    expected_len,), f"Expected input shape ({expected_len},), got {input_ids.shape}"
                assert target_ids.shape == (
                    expected_len,), f"Expected target shape ({expected_len},), got {target_ids.shape}"

                # Validate values are in vocab range
                assert input_ids.min() >= 0, f"Input has negative values: {input_ids.min()}"
                assert input_ids.max() < config[
                    'vocab_size'], f"Input exceeds vocab: {input_ids.max()} >= {config['vocab_size']}"
                assert target_ids.min() >= 0, f"Target has negative values: {target_ids.min()}"
                assert target_ids.max() < config[
                    'vocab_size'], f"Target exceeds vocab: {target_ids.max()} >= {config['vocab_size']}"

                # Log sample values
                sample_size = min(10, len(input_ids))
                logger.debug(f"    Sample input:  {input_ids[:sample_size].tolist()}")
                logger.debug(f"    Sample target: {target_ids[:sample_size].tolist()}")

            except Exception as e:
                logger.error(f"    Failed on pattern {pattern_name}: {e}")
                raise

    # Test smart data module
    logger.info("Testing Smart MockDataModule...")
    dm = create_smart_data_module(
        num_samples=50,
        sequence_length=32,
        vocab_size=100,
        batch_size=4
    )

    dm.setup("fit")

    # Test train dataloader
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    input_ids, target_ids = batch

    logger.info(f"DataModule batch shapes: input={input_ids.shape}, target={target_ids.shape}")
    expected_input_shape = (4, 31)  # batch_size=4, seq_len=32-1
    expected_target_shape = (4, 31)

    assert input_ids.shape == expected_input_shape, f"Expected {expected_input_shape}, got {input_ids.shape}"
    assert target_ids.shape == expected_target_shape, f"Expected {expected_target_shape}, got {target_ids.shape}"

    # Validate batch values
    assert input_ids.min() >= 0, f"Batch input has negative values"
    assert input_ids.max() < 100, f"Batch input exceeds vocab"
    assert target_ids.min() >= 0, f"Batch target has negative values"
    assert target_ids.max() < 100, f"Batch target exceeds vocab"

    logger.success("✅ Mock data test passed!")


if __name__ == "__main__":
    from loguru import logger

    test_mock_data()