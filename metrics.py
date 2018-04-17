import numpy as np
from tqdm import tqdm

from utils import decompose, get_logger

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


def compute_precision_at(ious, threshold):
    mx1 = np.max(ious, axis=0)
    mx2 = np.max(ious, axis=1)
    tp = np.sum(mx2 >= threshold)
    fp = np.sum(mx2 < threshold)
    fn = np.sum(mx1 < threshold)
    return float(tp) / (tp + fp + fn)


def compute_eval_metric(gt, predictions):
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    ious = compute_ious(gt, predictions)
    precisions = [compute_precision_at(ious, th) for th in thresholds]
    return sum(precisions) / len(precisions)


def intersection_over_union(y_true, y_pred):
    ious = []
    for y_t, y_p in tqdm(list(zip(y_true, y_pred))):
        iou = compute_ious(y_t, y_p)
        iou_mean = 1.0 * np.sum(iou) / iou.shape[0]
        ious.append(iou_mean)
    return np.mean(ious)


def intersection_over_union_thresholds(y_true, y_pred):
    iouts = []
    for y_t, y_p in tqdm(list(zip(y_true, y_pred))):
        iouts.append(compute_eval_metric(y_t, y_p))
    return np.mean(iouts)


def calculate_iou_matrix(ground_truth, proposals):
    mat = np.zeros([len(ground_truth),len(proposals)])
    used_proposals = []
    for i, gt in enumerate(ground_truth):
        for j,prop in enumerate(proposals):
            if j in used_proposals:
                continue
            iou_ = iou(gt,prop)
            mat[i,j]=iou_
            if iou_>0.5:
                used_proposals.append(j)
                break
    return mat
