from collections import OrderedDict
from enum import Enum
from typing import Any, Optional

import numpy as np
import torch


class MetricType(Enum):
    IOU = 0
    DICE = 1
    FSCORE = 2

class MMSegMetric:
    """
    Metrics computer by mmseg.

    num_classes (int): number of semantic classes/categories.
    ignore_index (int | list[int] | None): ignored indices when computing metrics.
    metrics (list[MetricType]): metrics type that are going to be computed. Currently provided: iou, dice, fscode.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int | list[int] | None = None,
        metrics: list[MetricType] = [MetricType.IOU],
        nan_to_num: Optional[int] = None,
    ) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.metrics = metrics
        self.nan_to_num = nan_to_num

        self.area_intersect = 0
        self.area_union = 0
        self.area_pred_label = 0
        self.area_label = 0

    def compute_and_accum(self, pred: torch.Tensor, label: torch.Tensor) -> None:
        """Compute and accumulate metrics' component.

        Args:
            pred (torch.Tensor): predicted index map.
            label (torch.Tensor): label index map.
        """
        area_intersect, area_union, area_pred_label, area_label = intersect_and_union(
            pred, label, self.num_classes, self.ignore_index
        )
        self.area_intersect += area_intersect
        self.area_union += area_union
        self.area_pred_label += area_pred_label
        self.area_label += area_label

    def get_and_clear(self) -> dict[str, np.ndarray]:
        """Get final result with accumulated component and clear all the accumulated component.

        Returns:
            dict[str, np.ndarray]: Metric results.
        """
        metrics = total_area_to_metrics(
            self.area_intersect,
            self.area_union,
            self.area_pred_label,
            self.area_label,
            self.metrics,
            self.nan_to_num,
        )
        self.area_intersect = 0
        self.area_union = 0
        self.area_pred_label = 0
        self.area_label = 0
        return metrics
    
    def show_result(self, metrics_result: dict[str, np.ndarray], class_name: list[str]) -> None:
        """
        Show result of metrics.
        Args:
            metrics_result: dict[str, np.ndarray]
            class_name: list[str]
        Returns:
            None
        """
        metrics_list = list(metrics_result.keys())
        metrics_list.remove('aAcc')
        metrics_value_list = list(metrics_result.values())
        metrics_value_list.remove(metrics_result['aAcc'])

        print('Metrics for each class:')
        print('-------------------------------')
        print('{:20}'.format('Class Name'), end='\t\t')
        for i, metric in enumerate(metrics_list):
            print('{:10}'.format(metric), end='\t')
            
        for i in range(len(class_name)):
            print('\n{:20}'.format(class_name[i]), end='\t')
            for j in range(len(metrics_list)):
                print('{:10.1f}'.format(metrics_value_list[j][i] * 100), end='\t')

        print('\n-------------------------------')
        print('{:20}'.format('mean'), end='\t')
        for i, metric in enumerate(metrics_list):
            print('{:10.1f}'.format(metrics_value_list[i].mean() * 100), end='\t')
        print('\naAcc: {:.1f} %'.format(metrics_result['aAcc'] * 100))


def intersect_and_union(
    pred_label: torch.tensor,
    label: torch.tensor,
    num_classes: int,
    ignore_index: int | list[int],
):
    """Calculate Intersection and Union.

    Args:
        pred_label (torch.tensor): Prediction segmentation map
            or predict result filename. The shape is (H, W).
        label (torch.tensor): Ground truth segmentation map
            or label filename. The shape is (H, W).
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.

    Returns:
        torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
        torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
        torch.Tensor: The prediction histogram on all classes.
        torch.Tensor: The ground truth histogram on all classes.
    """

    if isinstance(ignore_index, int) or ignore_index == None:
        mask = label != ignore_index
    elif isinstance(ignore_index, list):
        for index in ignore_index:
            mask = mask & (label != index)
    else:
        raise TypeError("Ignore index should be int or list[int]!")

    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1
    ).cpu()
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1
    ).cpu()
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1
    ).cpu()
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def total_area_to_metrics(
    total_area_intersect: np.ndarray,
    total_area_union: np.ndarray,
    total_area_pred_label: np.ndarray,
    total_area_label: np.ndarray,
    metrics: list[MetricType],
    nan_to_num: Optional[int] = None,
    beta: int = 1,
):
    """Calculate evaluation metrics
    Args:
        total_area_intersect (np.ndarray): The intersection of prediction
            and ground truth histogram on all classes.
        total_area_union (np.ndarray): The union of prediction and ground
            truth histogram on all classes.
        total_area_pred_label (np.ndarray): The prediction histogram on
            all classes.
        total_area_label (np.ndarray): The ground truth histogram on
            all classes.
        metrics (List[str] | str): Metrics to be evaluated, 'mIoU' and
            'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be
            replaced by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
    Returns:
        Dict[str, np.ndarray]: per category evaluation metrics,
            shape (num_classes, ).
    """

    def f_score(precision, recall, beta=1):
        """calculate the f-score value.

        Args:
            precision (float | torch.Tensor): The precision value.
            recall (float | torch.Tensor): The recall value.
            beta (int): Determines the weight of recall in the combined
                score. Default: 1.

        Returns:
            [torch.tensor]: The f-score value.
        """
        score = (
            (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
        )
        return score

    all_acc = total_area_intersect.sum() / total_area_label.sum()
    ret_metrics = OrderedDict({"aAcc": all_acc})
    for metric in metrics:
        if metric == MetricType.IOU:
            iou = total_area_intersect / total_area_union
            acc = total_area_intersect / total_area_label
            ret_metrics["IoU"] = iou
            ret_metrics["Acc"] = acc
        elif metric == MetricType.DICE:
            dice = 2 * total_area_intersect / (total_area_pred_label + total_area_label)
            acc = total_area_intersect / total_area_label
            ret_metrics["Dice"] = dice
            ret_metrics["Acc"] = acc
        elif metric == MetricType.FSCORE:
            precision = total_area_intersect / total_area_pred_label
            recall = total_area_intersect / total_area_label
            f_value = torch.tensor(
                [f_score(x[0], x[1], beta) for x in zip(precision, recall)]
            )
            ret_metrics["Fscore"] = f_value
            ret_metrics["Precision"] = precision
            ret_metrics["Recall"] = recall

    ret_metrics = {metric: value.numpy() for metric, value in ret_metrics.items()}
    if nan_to_num is not None:
        ret_metrics = OrderedDict(
            {
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            }
        )
    return ret_metrics

    