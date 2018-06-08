import numpy as np
from tqdm import tqdm

from .utils import decompose, get_logger

logger = get_logger()


def iou(gt, pred):
    gt[gt > 0] = 1.
    pred[pred > 0] = 1.
    intersection = gt * pred
    union = gt + pred
    union[union > 0] = 1.
    intersection = np.sum(intersection)
    union = np.sum(union)
    if union == 0:
        union = 1e-09
    return intersection / union


def compute_ious(gt, predictions):
    gt_ = decompose(gt)
    predictions_ = decompose(predictions)
    gt_ = np.asarray([el.flatten() for el in gt_])
    predictions_ = np.asarray([el.flatten() for el in predictions_])
    ious = calculate_iou_matrix(gt_, predictions_)
    return ious


def compute_precision_and_recall_at(ious, threshold):
    mx1 = np.max(ious, axis=0)
    mx2 = np.max(ious, axis=1)
    tp = np.sum(mx2 >= threshold)
    fp = np.sum(mx2 < threshold)
    fn = np.sum(mx1 < threshold)
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    return precision, recall


def compute_eval_metric(gt, predictions):
    threshold = 0.5
    ious = compute_ious(gt, predictions)
    return compute_precision_and_recall_at(ious, threshold)


def mean_precision_and_recall(y_true, y_pred):
    precisions = []
    recalls = []
    for y_t, y_p in tqdm(list(zip(y_true, y_pred))):
        precision, recall = compute_eval_metric(y_t, y_p)
        precisions.append(precision)
        recalls.append(recall)
    return np.mean(precisions), np.mean(recalls)


def calculate_iou_matrix(ground_truth, proposals):
    mat = np.zeros([len(ground_truth), len(proposals)])
    used_proposals = []
    for i, gt in enumerate(ground_truth):
        for j, prop in enumerate(proposals):
            if j in used_proposals:
                continue
            iou_ = iou(gt, prop)
            mat[i, j] = iou_
            if iou_ > 0.5:
                used_proposals.append(j)
                break
    return mat
