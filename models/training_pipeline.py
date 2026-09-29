import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import wandb
import numpy as np
import logging
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from omegaconf import DictConfig

from evaluation.trajectory_metrics import compute_mechanism_trajectory_metrics

from utils.device import (
    configure_torch_runtime,
    dataloader_kwargs,
    describe_device,
    get_device_request,
    resolve_device,
    use_non_blocking,
)

log = logging.getLogger(__name__)

class TrainingPipeline:
    """
    Standardizes training loop for the GNN models.
    """
    def __init__(
        self,
        config: DictConfig,
        model: nn.Module,
        dataset,
        checkpoint_dir=None,
        split_seed: int = 42,
    ):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.device = resolve_device(get_device_request(self.config))
        configure_torch_runtime(self.device)
        self.non_blocking = use_non_blocking(self.device)
        self.loader_kwargs = dataloader_kwargs(self.device)
        self.model.to(self.device)
        log.info("TrainingPipeline using %s", describe_device(self.device))
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.split_seed = int(split_seed)
        
        # Configure hyperparameters
        train_cfg = self.config.get('training', {})
        self.epochs = train_cfg.get('epochs', 100)
        self.batch_size = train_cfg.get('batch_size', 32)
        self.lr = train_cfg.get('learning_rate', train_cfg.get('lr', 1e-3))
        self.patience = train_cfg.get('patience', 10)
        self.weight_decay = train_cfg.get('weight_decay', 1e-4)
        
        # We need a proper Train/Val/Test split since the base dataset object doesn't do it automatically
        total_len = len(dataset)
        if total_len == 0:
            raise ValueError("TrainingPipeline requires a non-empty dataset")

        if total_len == 1:
            train_len, val_len, test_len = 1, 0, 0
        elif total_len == 2:
            train_len, val_len, test_len = 1, 0, 1
        else:
            train_len = max(int(0.8 * total_len), 1)
            val_len = max(int(0.1 * total_len), 1)
            test_len = total_len - train_len - val_len
            if test_len <= 0:
                test_len = 1
                if train_len > val_len:
                    train_len -= 1
                else:
                    val_len -= 1
        
        # For simplicity in Phase 2 unless specifically requested, we use random split
        generator = torch.Generator().manual_seed(self.split_seed)
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_len, val_len, test_len], generator=generator)
        
        self._train_loader_generator = torch.Generator().manual_seed(self.split_seed)
        self.train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            generator=self._train_loader_generator,
            **self.loader_kwargs,
        )
        self.val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, **self.loader_kwargs)
        self.test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, **self.loader_kwargs)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        
        # Handle loss based on model type (single step vs trajectory)
        if hasattr(self.model, 'trajectory_loss'):
            self.criterion = self.model.trajectory_loss
            self.is_trajectory = True
        else:
            self.criterion = nn.MSELoss()
            self.is_trajectory = False

        self.uses_x_history = bool(getattr(self.model, 'uses_x_history', False))
        default_target_name = 'y_trajectory' if self.is_trajectory else 'y'
        default_prefix = 'trajectory_' if self.is_trajectory else 'predictor_'
        self.target_name = getattr(self.model, 'target_name', default_target_name)
        self.checkpoint_prefix = getattr(self.model, 'checkpoint_prefix', default_prefix)

    def _forward_batch(self, batch):
        if self.uses_x_history:
            if not hasattr(batch, 'x_history'):
                raise ValueError("Temporal models require batch.x_history")
            return self.model(batch.x_history, batch.edge_index, batch.edge_attr, batch.batch)
        return self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

    def _target_batch(self, batch):
        if not hasattr(batch, self.target_name):
            raise ValueError(f"Model target '{self.target_name}' is missing from the batch")
        return getattr(batch, self.target_name)

    def _compute_loss(self, preds, target):
        if preds.shape != target.shape:
            raise ValueError(
                "Prediction and target shapes must match exactly; "
                f"got {tuple(preds.shape)} and {tuple(target.shape)}"
            )
        return self.criterion(preds, target)
            
    def train(self) -> dict:
        # For trajectory: monitor R² (higher=better). For predictor: monitor loss (lower=better).
        best_val_loss = float('-inf') if self.is_trajectory else float('inf')
        patience_counter = 0
        
        checkpoint_dir = self._checkpoint_dir if self._checkpoint_dir else Path("checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        prefix = self.checkpoint_prefix
        best_path = checkpoint_dir / f"{prefix}best.pt"
        last_path = checkpoint_dir / f"{prefix}last.pt"
        
        for epoch in range(self.epochs):
            # Training Phase
            self.model.train()
            train_loss = 0.0
            
            for batch in self.train_loader:
                batch = batch.to(self.device, non_blocking=self.non_blocking)
                self.optimizer.zero_grad()
                
                preds = self._forward_batch(batch)
                
                target = self._target_batch(batch)
                loss = self._compute_loss(preds, target)
                
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                train_loss += loss.item() * batch.num_graphs
                
            train_loss /= max(len(self.train_loader.dataset), 1)
            self.scheduler.step()
            
            # Validation Phase
            val_metrics = self.evaluate(split='val')
            val_loss = val_metrics['loss']
            
            if wandb.run is not None:
                try:
                    wandb.log({
                        f"train/{prefix}loss": train_loss,
                        f"val/{prefix}loss": val_loss,
                        f"val/{prefix}mae": val_metrics['mae'],
                        f"val/{prefix}r2": val_metrics['r2'],
                        "epoch": epoch
                    })
                except Exception as e:
                    log.warning(f"W&B logging failed: {e}")
                
        # Early Stopping and model saving — monitor R² for trajectory, loss for predictor
            monitor_r2 = self.is_trajectory
            if monitor_r2:
                metric_val = val_metrics['r2']
                improved = metric_val > best_val_loss  # best_val_loss reused as best_r2
            else:
                metric_val = val_loss
                improved = metric_val < best_val_loss

            if improved:
                best_val_loss = metric_val
                patience_counter = 0
                torch.save(self.model.state_dict(), best_path)
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
        # Save last
        torch.save(self.model.state_dict(), last_path)
        
        # Load best for final test eval
        self.load_checkpoint(best_path)
        test_metrics = self.evaluate(split='test')
        
        return test_metrics
        
    def evaluate(self, split: str = 'test') -> dict:
        self.model.eval()
        loader = self.val_loader if split == 'val' else self.test_loader
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device, non_blocking=self.non_blocking)
                preds = self._forward_batch(batch)
                target = self._target_batch(batch)
                
                loss = self._compute_loss(preds, target)
                total_loss += loss.item() * batch.num_graphs
                
                all_preds.append(preds.cpu().numpy())
                all_targets.append(target.cpu().numpy())
                
        if len(loader.dataset) == 0 or not all_preds:
            return {
                'loss': 0.0,
                'mae': 0.0,
                'rmse': 0.0,
                'r2': 0.0
            }

        avg_loss = total_loss / max(len(loader.dataset), 1)
        
        preds_np = np.concatenate(all_preds, axis=0)
        targets_np = np.concatenate(all_targets, axis=0)
        
        trajectory_metrics = None
        if preds_np.ndim == 3 and targets_np.ndim == 3:
            trajectory_metrics = compute_mechanism_trajectory_metrics(
                preds_np, targets_np
            )
            aggregate = trajectory_metrics['aggregate']
            mae = aggregate['mae']
            rmse = aggregate['rmse']
            r2 = aggregate['r2']
        else:
            metric_targets = targets_np.reshape(-1) if targets_np.ndim > 2 else targets_np
            metric_preds = preds_np.reshape(-1) if preds_np.ndim > 2 else preds_np
            mae = mean_absolute_error(metric_targets, metric_preds)
            rmse = np.sqrt(mean_squared_error(metric_targets, metric_preds))
            r2 = r2_score(metric_targets, metric_preds)

        metrics = {
            'loss': avg_loss,
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2)
        }
        if trajectory_metrics is not None:
            metrics['trajectory_metrics'] = trajectory_metrics
        return metrics
        
    def load_checkpoint(self, path: Path) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
