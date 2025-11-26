"""Rewritten Trainer (Option C - Full refactor)
- Modular functions: train_one_epoch, validate, test, optimize_threshold
- Proper checkpointing (by validation loss)
- Probability-based metrics for validation & threshold optimization
- Supports supervised and self-supervised modes (keeps original interfaces)
- Avoids duplicated computation and provides clean logging

Assumptions about the user's environment (kept compatible with original trainer):
- DataLoader yields (data, labels, aug1, aug2) for supervised/self-supervised modes
- `model(data)` returns (predictions, features) in supervised mode
- `model(aug)` and `temporal_contr_model(features1, features2)` remain available for self-supervised mode
- `logger` behaves like Python logging.Logger

Original trainer file path (for reference):
ORIGINAL_TRAINER_PATH = "/mnt/data/trainer.py"
"""

from typing import Tuple, Optional
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from models.loss import NTXentLoss
import matplotlib.pyplot as plt

# Keep this constant so the code can reference the user's original trainer file if needed
ORIGINAL_TRAINER_PATH = "/mnt/data/trainer.py"


def plot_training_metrics(train_f1, valid_f1, train_prec, valid_prec, train_rec, valid_rec, log_dir):
    epochs = range(1, len(train_f1) + 1)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_f1, 'b-', label='Train F1', linewidth=2)
    plt.plot(epochs, valid_f1, 'r-', label='Valid F1', linewidth=2)
    plt.title('F1 Score Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_prec, 'b-', label='Train Precision', linewidth=2)
    plt.plot(epochs, valid_prec, 'r-', label='Valid Precision', linewidth=2)
    plt.title('Precision Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_rec, 'b-', label='Train Recall', linewidth=2)
    plt.plot(epochs, valid_rec, 'r-', label='Valid Recall', linewidth=2)
    plt.title('Recall Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(log_dir, exist_ok=True)
    plot_path = os.path.join(log_dir, 'training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training metrics plot saved to: {plot_path}")
    plt.close()


def optimize_threshold_from_probs(all_probs: np.ndarray, all_labels: np.ndarray, n_steps: int = 100) -> Tuple[float, float]:
    """Return (best_threshold, best_f1) by sweeping thresholds between 0 and 1."""
    best_f1 = 0.0
    best_thr = 0.5
    thresholds = np.linspace(0.0, 1.0, n_steps)
    for thr in thresholds:
        preds = (all_probs > thr).astype(int)
        f1 = f1_score(all_labels, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)
    return best_thr, best_f1


def collect_probs_and_labels(model, dataloader, device, training_mode: str):
    """Run model on dataloader and return numpy arrays of probs for class 1 and labels.
    For supervised mode: model(data) -> (predictions, features)
    For self_supervised mode: returns empty arrays
    """
    model.eval()
    all_probs = []
    all_labels = []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for data, labels, aug1, aug2 in dataloader:
            data = data.float().to(device)
            labels = labels.long().to(device)
            if training_mode == 'self_supervised':
                continue
            output = model(data)
            # output could be (predictions, features) or just predictions depending on implementation
            if isinstance(output, tuple) or isinstance(output, list):
                predictions = output[0]
            else:
                predictions = output
            probs = softmax(predictions)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    if len(all_labels) == 0:
        return np.array([]), np.array([])
    return np.array(all_probs), np.array(all_labels)


def evaluate_with_thresholds(model, dataloader, device, threshold: float = 0.5, training_mode: str = 'supervised'):
    """Evaluate model using a fixed threshold. Returns loss, f1, precision, recall, probs, labels"""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_losses = []
    all_preds = []
    all_labels = []
    all_probs = []

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for data, labels, aug1, aug2 in dataloader:
            data = data.float().to(device)
            labels = labels.long().to(device)
            if training_mode == 'self_supervised':
                continue
            output = model(data)
            predictions = output[0] if (isinstance(output, tuple) or isinstance(output, list)) else output
            loss = criterion(predictions, labels)
            probs = softmax(predictions)[:, 1]
            preds = (probs > threshold).long()

            total_losses.append(loss.item())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = np.mean(total_losses) if len(total_losses) > 0 else 0.0
    if len(all_labels) > 0:
        f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
        prec = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
        rec = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    else:
        f1 = prec = rec = 0.0

    return avg_loss, f1, prec, rec, np.array(all_probs), np.array(all_labels)


def train_one_epoch(model, temporal_contr_model, optimizer_model, optimizer_temp, train_loader, device, config, training_mode: str):
    model.train()
    temporal_contr_model.train()

    all_preds = []
    all_labels = []
    total_losses = []

    criterion = nn.CrossEntropyLoss()

    accumulation_steps = getattr(config, 'accumulation_steps', 1)

    optimizer_model.zero_grad()
    if optimizer_temp is not None:
        optimizer_temp.zero_grad()

    for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):
        print(f"Training batch {batch_idx+1}/{len(train_loader)}", end='\r')
        data = data.float().to(device)
        labels = labels.long().to(device)
        aug1 = aug1.float().to(device)
        aug2 = aug2.float().to(device)

        if training_mode == 'self_supervised':
            # Keep original self-supervised behavior
            pred1, feat1 = model(aug1)
            pred2, feat2 = model(aug2)

            feat1 = F.normalize(feat1, dim=1)
            feat2 = F.normalize(feat2, dim=1)

            temp_loss1, _ = temporal_contr_model(feat1, feat2)
            temp_loss2, _ = temporal_contr_model(feat2, feat1)

            nt_xent = NTXentLoss(device, config.batch_size, config.Context_Cont.temperature, config.Context_Cont.use_cosine_similarity)
            loss = (temp_loss1 + temp_loss2) * 1.0 + nt_xent(feat1, feat2) * 0.7

        else:
            output = model(data)
            predictions = output[0] if (isinstance(output, tuple) or isinstance(output, list)) else output
            loss = criterion(predictions, labels)
            pred_classes = predictions.detach().argmax(dim=1)
            all_preds.extend(pred_classes.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # gradient accumulation
        loss = loss / accumulation_steps
        total_losses.append(loss.item() * accumulation_steps)
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            optimizer_model.step()
            if optimizer_temp is not None:
                optimizer_temp.step()
            optimizer_model.zero_grad()
            if optimizer_temp is not None:
                optimizer_temp.zero_grad()


    print(" " * 50)  # Clear the line after epoch

    avg_loss = float(np.mean(total_losses)) if len(total_losses) > 0 else 0.0
    if training_mode == 'self_supervised':
        train_f1 = train_prec = train_rec = 0.0
    else:
        train_f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0) if len(all_labels) > 0 else 0.0
        train_prec = precision_score(all_labels, all_preds, pos_label=1, zero_division=0) if len(all_labels) > 0 else 0.0
        train_rec = recall_score(all_labels, all_preds, pos_label=1, zero_division=0) if len(all_labels) > 0 else 0.0

    return avg_loss, train_f1, train_prec, train_rec


def validate(model, temporal_contr_model, valid_loader, device, training_mode: str):
    """Validate using probability-based threshold 0.5 for initial metrics, return validation loss & metrics and also raw probs/labels."""
    # Use evaluate_with_threshold to compute metrics at fixed threshold 0.5
    val_loss, val_f1, val_prec, val_rec, all_probs, all_labels = evaluate_with_thresholds(model, valid_loader, device, threshold=0.5, training_mode=training_mode)
    return val_loss, val_f1, val_prec, val_rec, all_probs, all_labels


def test_and_report(model, temporal_contr_model, test_loader, device, training_mode: str, threshold: float = 0.5):
    test_loss, test_f1, test_prec, test_rec, probs, labels = evaluate_with_thresholds(model, test_loader, device, threshold=threshold, training_mode=training_mode)
    return test_loss, test_f1, test_prec, test_rec, probs, labels


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, device: torch.device = torch.device('cpu')) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location=device)


def Trainer(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device, logger, config, experiment_log_dir, training_mode: str = 'supervised'):
    """Full refactor Trainer.
    - Checkpoints by validation LOSS (stable)
    - Uses probability-based validation metrics (threshold 0.5) for reporting
    - Optimizes threshold on validation probs after loading best checkpoint
    - Evaluates test with default and optimized thresholds
    """
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    # Trackers
    train_f1_scores, valid_f1_scores = [], []
    train_precisions, valid_precisions = [], []
    train_recalls, valid_recalls = [], []

    best_valid_loss = float('inf')
    best_epoch = 0

    saved_models_dir = os.path.join(experiment_log_dir, "saved_models")
    os.makedirs(saved_models_dir, exist_ok=True)

    num_epochs = config.num_epoch

    for epoch in range(1, num_epochs + 1):
        train_loss, train_f1, train_prec, train_rec = train_one_epoch(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, device, config, training_mode)

        # Validate and get probs/labels for further thresholding later
        val_loss, val_f1, val_prec, val_rec, val_probs, val_labels = validate(model, temporal_contr_model, valid_dl, device, training_mode)

        # Scheduler step
        if training_mode != 'self_supervised':
            scheduler.step(val_loss)

        train_f1_scores.append(train_f1)
        valid_f1_scores.append(val_f1)
        train_precisions.append(train_prec)
        valid_precisions.append(val_prec)
        train_recalls.append(train_rec)
        valid_recalls.append(val_rec)

        logger.debug(f'Epoch : {epoch}\n'
                     f'Train Loss: {train_loss:.4f} | F1: {train_f1:2.4f} | Precision: {train_prec:2.4f} | Recall: {train_rec:2.4f}\n'
                     f'Valid Loss: {val_loss:.4f} | F1: {val_f1:2.4f} | Precision: {val_prec:2.4f} | Recall: {val_rec:2.4f}\n')

        # Checkpoint selection based on validation loss (stable for unbalanced sets)
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            best_epoch = epoch
            logger.debug(f"New best valid loss: {best_valid_loss:.4f} at epoch {epoch}, saving model checkpoint.")
            chkpoint = {'model_state_dict': model.state_dict(), 'temporal_contr_model_state_dict': temporal_contr_model.state_dict(), 'epoch': epoch}
            save_checkpoint(chkpoint, os.path.join(saved_models_dir, 'ckp_best.pt'))

        # Save last
        chkpoint_last = {'model_state_dict': model.state_dict(), 'temporal_contr_model_state_dict': temporal_contr_model.state_dict(), 'epoch': epoch}
        save_checkpoint(chkpoint_last, os.path.join(saved_models_dir, 'ckp_last.pt'))

    # plot metrics
    if len(train_f1_scores) > 0:
        plot_training_metrics(train_f1_scores, valid_f1_scores, train_precisions, valid_precisions, train_recalls, valid_recalls, experiment_log_dir)

    # Load best checkpoint (by validation loss)
    logger.debug(f'Loading best model from epoch {best_epoch}')
    best_chk = load_checkpoint(os.path.join(saved_models_dir, 'ckp_best.pt'), device=device)
    if best_chk is None:
        logger.error('No checkpoint found at saved_models/ckp_best.pt')
        return

    model.load_state_dict(best_chk['model_state_dict'])
    temporal_contr_model.load_state_dict(best_chk['temporal_contr_model_state_dict'])
    model.to(device)
    temporal_contr_model.to(device)

    # Collect validation probs/labels for threshold optimization
    logger.debug('Collecting validation probabilities for threshold optimization...')
    val_probs, val_labels = collect_probs_and_labels(model, valid_dl, device, training_mode)

    if val_labels.size == 0:
        logger.debug('Validation labels empty — skipping threshold optimization.')
        optimal_threshold = 0.5
    else:
        optimal_threshold, best_val_f1 = optimize_threshold_from_probs(val_probs, val_labels, n_steps=200)
        logger.debug('Threshold optimization results:')
        logger.debug(f'Best threshold: {optimal_threshold:.3f}')
        logger.debug(f'Best validation F1: {best_val_f1:.4f}')
        logger.debug(f'Validation positive rate: {np.mean(val_labels)*100:.1f}%')

    # Evaluate on the test set with default and optimized thresholds
    logger.debug('Evaluating on test set:')
    logger.debug('1. With default threshold (0.5):')
    test_loss_default, test_f1_default, test_precision_default, test_recall_default, test_probs, test_labels = test_and_report(model, temporal_contr_model, test_dl, device, training_mode, threshold=0.5)
    logger.debug(f'Test Loss: {test_loss_default:0.4f} | F1: {test_f1_default:0.4f} | Precision: {test_precision_default:0.4f} | Recall: {test_recall_default:0.4f}')

    logger.debug(f'2. With optimized threshold ({optimal_threshold:.3f}):')
    test_loss_opt, test_f1_opt, test_precision_opt, test_recall_opt, _, _ = test_and_report(model, temporal_contr_model, test_dl, device, training_mode, threshold=optimal_threshold)
    logger.debug(f'Test Loss: {test_loss_opt:0.4f} | F1: {test_f1_opt:0.4f} | Precision: {test_precision_opt:0.4f} | Recall: {test_recall_opt:0.4f}')

    f1_improvement = test_f1_opt - test_f1_default
    logger.debug('=== THRESHOLD OPTIMIZATION RESULTS ===')
    logger.debug(f'F1 improvement: {f1_improvement:+.4f} (from {test_f1_default:.4f} to {test_f1_opt:.4f})')
    logger.debug(f'Optimal threshold: {optimal_threshold:.3f} (vs default 0.5)')
    logger.debug(f'Best validation loss epoch: {best_epoch}')

    logger.debug('################## Training is Done! #########################')

# === Compatibility shim ===
# Your main.py imports `model_evaluate` from this file.
# To avoid breaking the old pipeline, we provide a thin wrapper.
# It simply calls `test_and_report` with threshold=0.5.

def model_evaluate(model, temporal_contr_model, dataloader, device, training_mode='supervised'):
    """
    Compatibility wrapper for legacy code.
    Old code expects:
    loss, f1, prec, rec, pred_labels, true_labels
    So we return exactly that.
    """
    loss, f1, prec, rec, probs, labels = test_and_report(
    model,
    temporal_contr_model,
    dataloader,
    device,
    training_mode=training_mode,
    threshold=0.5,
    )
    # Convert probs->pred labels
    pred_labels = (probs > 0.5).astype(int)
    true_labels = labels
    return loss, f1, prec, rec, pred_labels, true_labels

