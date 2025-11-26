import os

# Fix OpenMP library conflict - must be set before importing torch or other libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch

import numpy as np
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
from utils import _logger, set_requires_grad
from dataloader.dataloader import data_generator
from trainer.trainer import Trainer, model_evaluate
from models.TC import TC
from utils import _calc_metrics, copy_Files
from models.model import base_Model
# Args selections
start_time = datetime.now()


def plot_transition_predictions(true_labels, pred_labels, experiment_log_dir, window_duration=10, stride_duration=1):
    """
    Plot transition predictions vs ground truth with detailed analysis
    
    Args:
        true_labels: Ground truth labels (0 or 1)
        pred_labels: Predicted labels (0 or 1)
        experiment_log_dir: Directory to save the plot
        window_duration: Duration of each window in seconds
        stride_duration: Stride between windows in seconds
    """
    # Calculate time axis (window end times)
    num_windows = len(true_labels)
    time_points = [(i * stride_duration + window_duration) for i in range(num_windows)]
    
    # Create classification categories
    true_positives = (true_labels == 1) & (pred_labels == 1)
    false_positives = (true_labels == 0) & (pred_labels == 1)
    false_negatives = (true_labels == 1) & (pred_labels == 0)
    true_negatives = (true_labels == 0) & (pred_labels == 0)
    
    # Create the plot
    plt.figure(figsize=(20, 10))
    
    # Main timeline plot
    plt.subplot(3, 1, 1)
    
    # Plot ground truth transitions
    true_transition_times = [time_points[i] for i in range(num_windows) if true_labels[i] == 1]
    for t in true_transition_times:
        plt.axvline(x=t, color='green', alpha=0.7, linewidth=2, label='True Transition' if t == true_transition_times[0] else "")
    
    # Plot predicted transitions with different colors for TP/FP
    tp_times = [time_points[i] for i in range(num_windows) if true_positives[i]]
    fp_times = [time_points[i] for i in range(num_windows) if false_positives[i]]
    
    for t in tp_times:
        plt.axvline(x=t, color='blue', alpha=0.8, linewidth=3, label='True Positive' if t == tp_times[0] else "")
    for t in fp_times:
        plt.axvline(x=t, color='red', alpha=0.8, linewidth=3, label='False Positive' if t == fp_times[0] else "")
    
    plt.title('Transition Predictions vs Ground Truth (Timeline View)', fontsize=16)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Transitions', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Detailed comparison plot
    plt.subplot(3, 1, 2)
    
    # Create binary signals for visualization
    y_offset = 0.1
    plt.plot(time_points, true_labels + y_offset, 'g-', linewidth=2, label='Ground Truth', alpha=0.7)
    plt.plot(time_points, pred_labels - y_offset, 'b-', linewidth=2, label='Predictions', alpha=0.7)
    
    # Highlight errors
    for i in range(num_windows):
        if false_positives[i]:
            plt.scatter(time_points[i], pred_labels[i] - y_offset, color='red', s=50, marker='x', label='False Positive' if i == np.where(false_positives)[0][0] else "")
        if false_negatives[i]:
            plt.scatter(time_points[i], true_labels[i] + y_offset, color='orange', s=50, marker='s', label='False Negative' if i == np.where(false_negatives)[0][0] else "")
    
    plt.title('Binary Signal Comparison', fontsize=16)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Label Value', fontsize=12)
    plt.ylim(-0.3, 1.3)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Confusion matrix visualization
    plt.subplot(3, 1, 3)
    
    # Calculate metrics
    tp_count = np.sum(true_positives)
    fp_count = np.sum(false_positives)
    fn_count = np.sum(false_negatives)
    tn_count = np.sum(true_negatives)
    
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Create confusion matrix as bar chart
    categories = ['True Positives', 'False Positives', 'False Negatives', 'True Negatives']
    counts = [tp_count, fp_count, fn_count, tn_count]
    colors = ['green', 'red', 'orange', 'gray']
    
    bars = plt.bar(categories, counts, color=colors, alpha=0.7)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.title(f'Confusion Matrix - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}', fontsize=16)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add text box with summary statistics
    stats_text = f'''Summary Statistics:
    Total Windows: {num_windows}
    True Transitions: {np.sum(true_labels)}
    Predicted Transitions: {np.sum(pred_labels)}
    
    True Positives: {tp_count}
    False Positives: {fp_count}
    False Negatives: {fn_count}
    True Negatives: {tn_count}'''
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(experiment_log_dir, 'transition_predictions_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nTransition analysis plot saved to: {plot_path}")
    
    # Display the plot
    plt.show()
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp_count,
        'fp': fp_count,
        'fn': fn_count,
        'tn': tn_count
    }


parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Exp1', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=0, type=int,
                    help='seed value')
parser.add_argument('--training_mode', default='supervised', type=str,
                    help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear')
parser.add_argument('--selected_dataset', default='Epilepsy', type=str,
                    help='Dataset of choice: sleepEDF, HAR, Epilepsy, pFD')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
args = parser.parse_args()



device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
method = 'TS-TCC'
training_mode = args.training_mode
run_description = args.run_description

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)


exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs()

# ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################

experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, training_mode + f"_seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains
counter = 0
src_counter = 0


# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Method:  {method}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

# Load datasets
data_path = f"./data/{data_type}"
train_dl, valid_dl, test_dl = data_generator(data_path, configs, training_mode)
logger.debug("Data loaded ...")

# Load Model
model = base_Model(configs).to(device)
temporal_contr_model = TC(configs, device).to(device)

if training_mode == "fine_tune":
    # load saved model of this experiment
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}", "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_best.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = model.state_dict()
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

if training_mode == "train_linear" or "tl" in training_mode:
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}", "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_best.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # delete these parameters (Ex: the linear layer at the end)
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    set_requires_grad(model, pretrained_dict, requires_grad=False)  # Freeze everything except last layer.

if training_mode == "random_init":
    model_dict = model.state_dict()

    # delete all the parameters except for logits
    del_list = ['logits']
    pretrained_dict_copy = model_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del model_dict[i]
    set_requires_grad(model, model_dict, requires_grad=False)  # Freeze everything except last layer.



model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

if training_mode == "self_supervised":  # to do it only once
    copy_Files(os.path.join(logs_save_dir, experiment_description, run_description), data_type)

# Trainer
Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl, valid_dl, test_dl, device, logger, configs, experiment_log_dir, training_mode)

if training_mode != "self_supervised":
    # Testing
    outs = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
    total_loss, total_f1, total_precision, total_recall, pred_labels, true_labels = outs
    _calc_metrics(pred_labels, true_labels, experiment_log_dir, args.home_path)
    
    # Create detailed transition analysis plot
    logger.debug("Creating transition analysis visualization...")
    analysis_results = plot_transition_predictions(true_labels, pred_labels, experiment_log_dir)
    logger.debug(f"Analysis Results: Precision={analysis_results['precision']:.3f}, "
                f"Recall={analysis_results['recall']:.3f}, F1={analysis_results['f1']:.3f}")
    logger.debug(f"TP={analysis_results['tp']}, FP={analysis_results['fp']}, "
                f"FN={analysis_results['fn']}, TN={analysis_results['tn']}")

logger.debug(f"Training time is : {datetime.now()-start_time}")
