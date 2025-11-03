import torch

def bbox_overlaps_torch(bboxes1: torch.Tensor,
                        bboxes2: torch.Tensor,
                        mode: str = 'iou',
                        eps: float = 1e-6) -> torch.Tensor:
    """
    PyTorch version of bbox_overlaps.
    
    Args:
        bboxes1 (Tensor): shape (N, 4) in (x1, y1, x2, y2) format
        bboxes2 (Tensor): shape (M, 4) in (x1, y1, x2, y2) format
        mode (str): 'iou' (intersection over union) or 'iof' (intersection over foreground)
        eps (float): small value to avoid divide by zero
    
    Returns:
        Tensor: shape (N, M), overlaps between bboxes1 and bboxes2
    """
    assert mode in ['iou', 'iof']
    N = bboxes1.size(0)
    M = bboxes2.size(0)

    # areas
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]).clamp(min=0) * \
            (bboxes1[:, 3] - bboxes1[:, 1]).clamp(min=0)  # (N,)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]).clamp(min=0) * \
            (bboxes2[:, 3] - bboxes2[:, 1]).clamp(min=0)  # (M,)

    # intersections
    lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])   # (N, M, 2)
    rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])   # (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    if mode == 'iou':
        union = area1[:, None] + area2 - inter
    else:  # 'iof' -> intersection over area1
        union = area1[:, None]

    return inter / (union + eps)
