"""Final tracker: update every valid LK point even when only one remains.

Legacy evaluator SHA256: 31f70f2eed133e93848357bbe8952fdb76ea865d48a202086c824f42d3c0a0f8
Only the valid-point update placement changed; bbox transform still needs two points.
"""
def propagate(previous_gray, gray, boxes, points, valid, return_details=False):
    import cv2
    import numpy as np
    if not len(points):
        result=boxes.copy(), points.copy(), valid.copy(), float('inf'), False
        return (*result,{'failed':True}) if return_details else result
    original = points.astype(np.float32).reshape(-1, 1, 2)
    nxt, s1, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, original, None, winSize=(21, 21), maxLevel=3)
    if nxt is None:
        result=boxes.copy(), points.copy(), np.zeros_like(valid), float('inf'), False
        return (*result,{'failed':True}) if return_details else result
    back, s2, _ = cv2.calcOpticalFlowPyrLK(gray, previous_gray, nxt, None, winSize=(21, 21), maxLevel=3)
    if back is None:
        result=boxes.copy(), points.copy(), np.zeros_like(valid), float('inf'), False
        return (*result,{'failed':True}) if return_details else result
    error = np.linalg.norm(back[:, 0] - original[:, 0], axis=1).reshape(-1, 4)
    proposed = nxt.reshape(-1, 4, 2)
    inside = ((proposed[..., 0] >= 0) & (proposed[..., 0] < gray.shape[1]) &
              (proposed[..., 1] >= 0) & (proposed[..., 1] < gray.shape[0]))
    good = (valid & s1.reshape(-1, 4).astype(bool) & s2.reshape(-1, 4).astype(bool)
            & np.isfinite(proposed).all(axis=2) & (error <= 2.0) & inside)
    out_boxes, out_points = boxes.copy(), points.copy()
    if return_details:
        fit_ok=np.zeros(len(points),dtype=bool)
        fit_residual=np.full(len(points),np.inf)
    # Fit a simple motion transform per gate, preserving detector box margins.
    # Invalid output points are excluded from evaluation, never filled from GT.
    for j in range(len(points)):
        if good[j].sum() >= 2:
            matrix, _ = cv2.estimateAffinePartial2D(points[j, good[j]], proposed[j, good[j]], method=cv2.LMEDS)
            if matrix is not None and np.isfinite(matrix).all():
                out_points[j] = cv2.transform(points[j][None].astype(np.float32), matrix)[0]
                if return_details:
                    fit_ok[j]=abs(np.linalg.det(matrix[:,:2]))>1e-6
                    fit_residual[j]=np.median(np.linalg.norm(out_points[j,good[j]]-proposed[j,good[j]],axis=1))
                x1, y1, x2, y2 = boxes[j]
                corners = np.array([[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]], dtype=np.float32)
                warped = cv2.transform(corners, matrix)[0]
                out_boxes[j] = [*warped.min(axis=0), *warped.max(axis=0)]
        out_points[j, good[j]] = proposed[j, good[j]]
    reliable = bool(np.all(good.sum(axis=1) >= 3))
    max_error = float(error[valid].max()) if valid.any() else float('inf')
    result=out_boxes, out_points, good, max_error, reliable
    if return_details:
        return (*result,dict(failed=False,fb_errors=error,previous_valid=valid.copy(),fit_ok=fit_ok,fit_residual=fit_residual))
    return result
