import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
import numpy as np
from typing import Dict, List, Optional, Union, Iterator
import os
import json
import random
from pathlib import Path


class LLMPretrainingDataset(Dataset):
    """
    Dataset for LLM pretraining using popular open-source datasets.
    Supports OpenWebText, C4, The Pile, and other text datasets.
    """
    
    def __init__(
        self,
        dataset_name: str = "openwebtext",
        tokenizer_name: str = "gpt2",
        max_length: int = 1024,
        cache_dir: Optional[str] = None,
        streaming: bool = False,
        num_proc: int = 4,
        split: str = "train",
        subset: Optional[str] = None,
        text_column: str = "text",
        min_length: int = 10,
        **kwargs
    ):
        """
        Args:
            dataset_name: Name of the dataset to use
            tokenizer_name: Name or path of the tokenizer
            max_length: Maximum sequence length
            cache_dir: Directory to cache processed data
            streaming: Whether to use streaming mode for large datasets
            num_proc: Number of processes for data processing
            split: Dataset split to use
            subset: Subset of the dataset (if applicable)
            text_column: Name of the text column in the dataset
            min_length: Minimum text length to include
        """
        super().__init__()
        
        self.dataset_name = dataset_name
        self.max_length = max_length
        self.text_column = text_column
        self.min_length = min_length
        self.streaming = streaming
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load dataset
        self.dataset = self._load_dataset(
            dataset_name, cache_dir, streaming, num_proc, split, subset, **kwargs
        )
        
        if not streaming:
            # Preprocess and tokenize the entire dataset
            self.dataset = self._preprocess_dataset(num_proc)
    
    def _load_dataset(
        self, 
        dataset_name: str, 
        cache_dir: Optional[str], 
        streaming: bool, 
        num_proc: int, 
        split: str,
        subset: Optional[str],
        **kwargs
    ):
        """Load the specified dataset."""
        
        dataset_configs = {
            "openwebtext": {
                "path": "openwebtext",
                "name": None,
                "split": split
            },
            "c4": {
                "path": "c4",
                "name": "en",
                "split": split
            },
            "pile": {
                "path": "EleutherAI/pile",
                "name": None,
                "split": split
            },
            "bookcorpus": {
                "path": "bookcorpus",
                "name": None,
                "split": split
            },
            "wikipedia": {
                "path": "wikipedia",
                "name": "20220301.en",
                "split": split
            },
            "oscar": {
                "path": "oscar",
                "name": "unshuffled_deduplicated_en",
                "split": split
            },
            "redpajama": {
                "path": "togethercomputer/RedPajama-Data-1T",
                "name": None,
                "split": split
            }
        }
        
        if dataset_name not in dataset_configs:
            raise ValueError(f"Dataset {dataset_name} not supported. Available: {list(dataset_configs.keys())}")
        
        config = dataset_configs[dataset_name]
        config_name = subset or config["name"]
        
        try:
            print(f"Loading dataset: {dataset_name}")
            dataset = load_dataset(
                config["path"],
                name=config_name,
                split=config["split"],
                cache_dir=cache_dir,
                streaming=streaming,
                **kwargs
            )
            print(f"Dataset loaded successfully!")
            return dataset
            
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            print("Falling back to a smaller sample dataset...")
            # Fallback to a smaller, more reliable dataset
            return load_dataset("wikitext", "wikitext-2-raw-v1", split=split, streaming=streaming)
    
    def _preprocess_dataset(self, num_proc: int):
        """Preprocess and tokenize the dataset."""
        print("Preprocessing and tokenizing dataset...")
        
        def tokenize_function(examples):
            # Filter out short texts
            texts = [text for text in examples[self.text_column] 
                    if len(text.strip()) >= self.min_length]
            
            if not texts:
                return {"input_ids": [], "attention_mask": []}
            
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_attention_mask=True,
                add_special_tokens=True
            )
            
            return tokenized
        
        # Apply tokenization
        tokenized_dataset = self.dataset.map(
            tokenize_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=self.dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        # Filter out empty sequences
        tokenized_dataset = tokenized_dataset.filter(
            lambda x: len(x["input_ids"]) > 0,
            num_proc=num_proc
        )
        
        print(f"Dataset preprocessing complete. Total samples: {len(tokenized_dataset)}")
        return tokenized_dataset
    
    def __len__(self):
        if self.streaming:
            # For streaming datasets, we can't know the exact length
            return float('inf')
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.streaming:
            raise NotImplementedError("Use streaming dataset with DataLoader")
        
        item = self.dataset[idx]
        
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(item["attention_mask"], dtype=torch.long)
        
        # For causal LM, labels are the same as input_ids
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class StreamingLLMDataset(IterableDataset):
    """
    Streaming version for very large datasets that don't fit in memory.
    """
    
    def __init__(
        self,
        dataset_name: str = "c4",
        tokenizer_name: str = "gpt2",
        max_length: int = 1024,
        buffer_size: int = 1000,
        shuffle_buffer_size: int = 10000,
        text_column: str = "text",
        min_length: int = 10,
        **kwargs
    ):
        super().__init__()
        
        self.max_length = max_length
        self.text_column = text_column
        self.min_length = min_length
        self.buffer_size = buffer_size
        self.shuffle_buffer_size = shuffle_buffer_size
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load streaming dataset
        self.dataset = self._load_streaming_dataset(dataset_name, **kwargs)
    
    def _load_streaming_dataset(self, dataset_name: str, **kwargs):
        """Load dataset in streaming mode."""
        
        dataset_configs = {
            "c4": ("c4", "en"),
            "pile": ("EleutherAI/pile", None),
            "redpajama": ("togethercomputer/RedPajama-Data-1T", None),
            "oscar": ("oscar", "unshuffled_deduplicated_en")
        }
        
        if dataset_name in dataset_configs:
            path, name = dataset_configs[dataset_name]
            return load_dataset(path, name=name, split="train", streaming=True, **kwargs)
        else:
            # Fallback
            return load_dataset("c4", "en", split="train", streaming=True)
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over the dataset."""
        
        buffer = []
        
        for example in self.dataset:
            text = example[self.text_column]
            
            # Skip short texts
            if len(text.strip()) < self.min_length:
                continue
            
            # Tokenize
            tokenized = self.tokenizer(
                text,
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_attention_mask=True,
                add_special_tokens=True
            )
            
            if len(tokenized["input_ids"]) == 0:
                continue
            
            input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(tokenized["attention_mask"], dtype=torch.long)
            labels = input_ids.clone()
            
            item = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
            
            buffer.append(item)
            
            # Yield from buffer when it's full
            if len(buffer) >= self.buffer_size:
                # Shuffle buffer
                random.shuffle(buffer)
                for buffered_item in buffer:
                    yield buffered_item
                buffer = []
        
        # Yield remaining items in buffer
        random.shuffle(buffer)
        for item in buffer:
            yield item


def create_pretraining_dataloader(
    dataset_name: str = "openwebtext",
    tokenizer_name: str = "gpt2",
    max_length: int = 1024,
    batch_size: int = 8,
    streaming: bool = False,
    num_workers: int = 4,
    shuffle: bool = True,
    **dataset_kwargs
) -> DataLoader:
    """
    Create a DataLoader for LLM pretraining.
    
    Args:
        dataset_name: Name of the dataset
        tokenizer_name: Tokenizer to use
        max_length: Maximum sequence length
        batch_size: Batch size
        streaming: Use streaming dataset
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle the data
        **dataset_kwargs: Additional arguments for dataset
    
    Returns:
        DataLoader for pretraining
    """
    
    def collate_fn(batch):
        """Custom collate function for variable length sequences."""
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]
        
        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=0
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    if streaming:
        dataset = StreamingLLMDataset(
            dataset_name=dataset_name,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            **dataset_kwargs
        )
        # For streaming datasets, shuffle is handled internally
        shuffle = False
    else:
        dataset = LLMPretrainingDataset(
            dataset_name=dataset_name,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            streaming=False,
            **dataset_kwargs
        )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )


# Example usage and testing
def test_dataset():
    """Test the dataset implementation."""
    print("Testing LLM Pretraining Dataset...")
    
    # Test with a small dataset first
    try:
        # Use WikiText-2 as a reliable test dataset
        dataloader = create_pretraining_dataloader(
            dataset_name="wikitext",
            tokenizer_name="gpt2",
            max_length=512,
            batch_size=2,
            streaming=False
        )
        
        print("Loading a test batch...")
        batch = next(iter(dataloader))
        
        print(f"Batch keys: {batch.keys()}")
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        
        print("Dataset test successful!")
        
    except Exception as e:
        print(f"Error testing dataset: {e}")


if __name__ == "__main__":
    test_dataset()


# Configuration for different popular datasets
DATASET_CONFIGS = {
    "small_test": {
        "dataset_name": "wikitext",
        "subset": "wikitext-2-raw-v1",
        "max_length": 512,
        "batch_size": 4
    },
    "openwebtext": {
        "dataset_name": "openwebtext", 
        "max_length": 1024,
        "batch_size": 8
    },
    "c4_large": {
        "dataset_name": "c4",
        "max_length": 1024,
        "batch_size": 8,
        "streaming": True
    },
    "pile": {
        "dataset_name": "pile",
        "max_length": 2048,
        "batch_size": 4,
        "streaming": True
    },
    "redpajama": {
        "dataset_name": "redpajama",
        "max_length": 2048,
        "batch_size": 4,
        "streaming": True
    }
}