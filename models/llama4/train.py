from numpy.matlib import place
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from model import Transformer
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Optional, Dict, Any, Tuple
import torchmetrics
from pytorch_lightning.cli import LightningCLI

from models.llama4.args import ModelArgs


class LightningTransformer(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: Optional[int] = None,
        loss_fn: Optional[nn.Module] = None,
        metrics: Optional[Dict[str, Any]] = None,
        gradient_clip_val: float = 1.0,
        **kwargs
    ):
        """
        Lightning wrapper for transformer models.
        
        Args:
            model: The transformer model (nn.Module)
            learning_rate: Peak learning rate
            weight_decay: Weight decay for AdamW optimizer
            warmup_steps: Number of warmup steps for learning rate schedule
            max_steps: Maximum training steps (if None, will be set based on trainer)
            loss_fn: Loss function (defaults to CrossEntropyLoss)
            metrics: Dictionary of metrics to track
            gradient_clip_val: Gradient clipping value
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'loss_fn', 'metrics'])
        
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.gradient_clip_val = gradient_clip_val
        
        # Default loss function
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        
        # Initialize metrics
        self.train_metrics = nn.ModuleDict()
        self.val_metrics = nn.ModuleDict()
        self.test_metrics = nn.ModuleDict()
        
        if metrics:
            for name, metric_class in metrics.items():
                self.train_metrics[name] = metric_class()
                self.val_metrics[name] = metric_class()
                self.test_metrics[name] = metric_class()
    
    def forward(self, *args, **kwargs):
        """Forward pass through the model."""
        return self.model(*args, **kwargs)
    
    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        """Training step."""
        # Assume batch is (input_ids, labels) or similar
        if len(batch) == 2:
            inputs, labels = batch
            outputs = self(inputs)
        else:
            # More flexible handling for different batch formats
            *inputs, labels = batch
            outputs = self(*inputs)
        
        # Calculate loss
        if hasattr(outputs, 'logits'):
            # Handle model outputs with logits attribute (like HuggingFace models)
            logits = outputs.logits
        else:
            # Assume outputs are logits directly
            logits = outputs
        
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Log loss
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Calculate and log metrics
        if self.train_metrics:
            preds = torch.argmax(logits, dim=-1)
            for name, metric in self.train_metrics.items():
                metric_val = metric(preds.view(-1), labels.view(-1))
                self.log(f'train/{name}', metric_val, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        if len(batch) == 2:
            inputs, labels = batch
            outputs = self(inputs)
        else:
            *inputs, labels = batch
            outputs = self(*inputs)
        
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Log validation loss
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Calculate and log metrics
        if self.val_metrics:
            preds = torch.argmax(logits, dim=-1)
            for name, metric in self.val_metrics.items():
                metric_val = metric(preds.view(-1), labels.view(-1))
                self.log(f'val/{name}', metric_val, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        """Test step."""
        if len(batch) == 2:
            inputs, labels = batch
            outputs = self(inputs)
        else:
            *inputs, labels = batch
            outputs = self(*inputs)
        
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Log test loss
        self.log('test/loss', loss, on_step=False, on_epoch=True)
        
        # Calculate and log metrics
        if self.test_metrics:
            preds = torch.argmax(logits, dim=-1)
            for name, metric in self.test_metrics.items():
                metric_val = metric(preds.view(-1), labels.view(-1))
                self.log(f'test/{name}', metric_val, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Separate parameters for weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        if self.max_steps is None:
            # Try to estimate max_steps from trainer
            if self.trainer is not None:
                self.max_steps = self.trainer.estimated_stepping_batches
            else:
                # Fallback to a reasonable default
                self.max_steps = 10000
        
        # Learning rate scheduler with warmup
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.warmup_steps
        )
        
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_steps - self.warmup_steps,
            eta_min=0.1 * self.learning_rate
        )
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps]
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        """Configure gradient clipping."""
        if gradient_clip_val is not None and gradient_clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=gradient_clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm
            )


class TransformerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_path: str,
        val_data_path: str,
        batch_size: int = 32,
        max_seq_len: int = 2048,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
    
    def setup(self, stage: str):
        # Implement your data loading logic here
        # This is a placeholder - replace with your actual data loading
        pass
    
    def train_dataloader(self):
        # Return your training DataLoader
        # This is a placeholder
        return DataLoader(
            dataset=DummyDataset(1000, self.max_seq_len),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        # Return your validation DataLoader
        return DataLoader(
            dataset=DummyDataset(100, self.max_seq_len),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# Dummy dataset for example
class DummyDataset(Dataset):
    def __init__(self, size: int, seq_len: int, vocab_size: int = 32000):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.randint(0, self.vocab_size, (self.seq_len,)),
            'labels': torch.randint(0, self.vocab_size, (self.seq_len,))
        }


# Custom CLI class to handle ModelArgs
class TransformerCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # Add custom model arguments
        parser.add_class_arguments(ModelArgs, "model_args")
        parser.set_defaults({
            "model.backbone": lazy_instance(MyModel, encoder_layers=24)
        })
        # Link model_args to the model
        parser.link_arguments("model_args", "model.init_args.model_args")



# Example usage and helper function
def create_lightning_transformer(
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    max_steps: Optional[int] = None,
    task_type: str = "classification",
    num_classes: Optional[int] = None,
    **kwargs
) -> LightningTransformer:
    """
    Helper function to create a Lightning transformer with common configurations.
    
    Args:
        model: Your transformer model
        learning_rate: Peak learning rate
        weight_decay: Weight decay for AdamW
        warmup_steps: Warmup steps for LR schedule
        max_steps: Maximum training steps
        task_type: Type of task ("classification", "regression", "generation")
        num_classes: Number of classes for classification tasks
        **kwargs: Additional arguments for LightningTransformer
    
    Returns:
        LightningTransformer instance
    """
    
    # Configure loss function based on task type
    if task_type == "classification":
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    elif task_type == "regression":
        loss_fn = nn.MSELoss()
    elif task_type == "generation":
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    else:
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Configure metrics based on task type
    metrics = {}
    if task_type == "classification" and num_classes:
        metrics["accuracy"] = lambda: torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, ignore_index=-100
        )
        metrics["f1"] = lambda: torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, ignore_index=-100
        )
    
    return LightningTransformer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        loss_fn=loss_fn,
        metrics=metrics,
        **kwargs
    )


if __name__ == "__main__":
    cli = TransformerCLI(
        LightningTransformer,
        TransformerDataModule,
        seed_everything_default=42,
    )

    # args = ModelArgs()
    # model = Transformer(args)
    # lightning_model = create_lightning_transformer(
    # model=model,
    # learning_rate=5e-5,
    # warmup_steps=500,
    # task_type="classification",
    # num_classes=10
    # )

    # trainer = pl.Trainer(
    # max_epochs=10,
    # gradient_clip_val=1.0,
    # precision=16,  # Mixed precision
    # accelerator="gpu",
    # devices=1)

    # trainer.fit(lightning_model)

