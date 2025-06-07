# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.registry import HOOKS
from mmaction.registry import RUNNERS


def parse_args():
    parser = argparse.ArgumentParser(description='Train an action recognizer')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it; if not, '
             'auto resume from latest checkpoint in work dir.')
    parser.add_argument(
        '--amp',
        action='store_true',
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='auto scale the learning rate according to batch size')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff-rank-seed',
        action='store_true',
        help='set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the config, format key=val')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_args(cfg, args):
    """Merge CLI arguments into cfg."""
    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None

    cfg.launcher = args.launcher


    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join(
            './work_dirs/pull ups', osp.splitext(osp.basename(args.config))[0])

    if args.amp:
        optim_wrapper_type = cfg.optim_wrapper.get('type', 'OptimWrapper')
        assert optim_wrapper_type in ['OptimWrapper', 'AmpOptimWrapper'], \
            f'`--amp` not supported by optimizer wrapper `{optim_wrapper_type}`'
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    if args.auto_scale_lr:
        cfg.auto_scale_lr.enable = True

    if cfg.get('randomness', None) is None:
        cfg.randomness = dict(
            seed=args.seed,
            diff_rank_seed=args.diff_rank_seed,
            deterministic=args.deterministic)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg


def collect_predictions(model, data_loader, device):
    """모델을 이용해 예측값(predictions)과 실제값(labels)을 수집."""
    predictions = []
    labels = []

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            # 디버깅 정보 출력
            if i == 0:  # 첫 번째 샘플에 대해서만 상세 정보 출력
                print("\nDebugging information for first sample:")
                sample = data[0]
                print(f"Sample keys: {sample.keys()}")
                print(f"Inputs shape: {sample['inputs'].shape if isinstance(sample['inputs'], torch.Tensor) else [x.shape for x in sample['inputs']]}")
                print(f"Label info: {sample['data_samples'].label}")
                print(f"Device being used: {device}")
                
            sample = data[0]
            inputs = sample['inputs']
            if isinstance(inputs, list):
                # 입력이 리스트인 경우 처리 과정 출력
                print(f"\nSample {i}: Converting list of inputs to tensor")
                print(f"Input list length: {len(inputs)}")
                print(f"Each input shape: {inputs[0].shape}")
                inputs = torch.stack(inputs).to(device)
            else:
                inputs = inputs.to(device)
            
            label = int(sample['data_samples'].label.item())
            
            # 예측 과정 디버깅
            result = model.forward(inputs)
            pred_scores = result['pred_score']
            pred = pred_scores.argmax(dim=1).cpu().numpy()
            
            if i == 0:  # 첫 번째 샘플의 예측 상세 정보
                print(f"\nPrediction scores shape: {pred_scores.shape}")
                print(f"Raw prediction scores: {pred_scores.softmax(dim=1)}")
                print(f"Predicted class: {pred[0]}, True label: {label}")
            
            predictions.extend(pred)
            labels.append(label)

            # 진행 상황 표시
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} samples...")

    predictions = np.array(predictions, dtype=int)
    labels = np.array(labels, dtype=int)
    
    # 최종 통계 출력
    print("\nPrediction statistics:")
    print(f"Total samples processed: {len(predictions)}")
    print(f"Unique predicted classes: {np.unique(predictions)}")
    print(f"Class distribution in predictions: {np.bincount(predictions)}")
    print(f"Class distribution in true labels: {np.bincount(labels)}")
    
    return predictions, labels


def visualize_confusion_matrix(labels, predictions, save_dir):
    """혼동 행렬을 계산하고 시각화."""
    try:
        # 혼동 행렬 계산
        cm = confusion_matrix(labels, predictions)
        num_classes = cm.shape[0]
        
        # 정규화된 혼동 행렬 계산
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        print("\nConfusion Matrix Statistics:")
        print(f"Matrix shape: {cm.shape}")
        print(f"Total samples per class: {cm.sum(axis=1)}")
        print(f"Correctly classified samples per class: {cm.diagonal()}")
        print(f"Classification accuracy per class: {cm.diagonal() / cm.sum(axis=1)}")
        
        # 행렬 시각화
        plt.figure(figsize=(12, 8))
        
        # 원본 혼동 행렬
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (Raw Counts)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # 정규화된 혼동 행렬
        plt.subplot(1, 2, 2)
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
        plt.title('Confusion Matrix (Normalized)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # 저장
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'confusion_matrix.png')
        plt.savefig(save_path)
        print(f"\nConfusion matrix saved to: {save_path}")
        
        # 성능 지표 계산
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='weighted')
        recall = recall_score(labels, predictions, average='weighted')
        f1 = f1_score(labels, predictions, average='weighted')
        
        print("\nPerformance Metrics:")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Weighted Precision: {precision:.4f}")
        print(f"Weighted Recall: {recall:.4f}")
        print(f"Weighted F1-Score: {f1:.4f}")
        
    except Exception as e:
        print(f"\nError in visualize_confusion_matrix: {str(e)}")
        print(f"Labels shape: {labels.shape}, unique values: {np.unique(labels)}")
        print(f"Predictions shape: {predictions.shape}, unique values: {np.unique(predictions)}")
        raise


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg = merge_args(cfg, args)

    # Runner 생성
    if 'runner_type' in cfg:
        runner = RUNNERS.build(cfg)
    else:
        runner = Runner.from_cfg(cfg)

    # 훈련 시작
    runner.train()


if __name__ == '__main__':
    main()
