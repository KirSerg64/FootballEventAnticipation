from __future__ import annotations
import numpy as np
import supervision as sv
import torch


class SAM2Tracker:
    def __init__(self, predictor) -> None:
        self._predictor = predictor
        self._prompted = False
        self._track_id = []
        self._frame_idx = 0

    def prompt_first_frame(self, frame: np.ndarray, detections: sv.Detections) -> None:
        if len(detections.xyxy) == 0:
            raise ValueError("detections must contain at least one box")

        if not self._track_id:
            self._track_id = list(range(1, len(detections.xyxy) + 1))

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self._predictor.load_first_frame(frame)
            for bbox, obj_id in zip(detections.xyxy, self._track_id):
                _, out_obj_ids, out_mask_logits = self._predictor.add_new_prompt(
                    frame_idx=self._frame_idx,
                    obj_id=int(obj_id),
                    bbox=bbox,
                )

        self._prompted = True
        return None

    def track(
        self,
        frame: np.ndarray,
        new_detections: sv.Detections | None = None,
    ) -> sv.Detections:
        """Track all objects in *frame* and optionally seed new ones.

        Parameters
        ----------
        frame:
            Current BGR video frame.
        new_detections:
            Optional ``sv.Detections`` of newly found objects (e.g. from
            periodic re-detection) to add to SAM2 at the current frame before
            tracking.  Their internal SAM2 IDs are appended to ``_track_id``.
        """
        if not self._prompted:
            raise RuntimeError("Call prompt_first_frame before propagate")

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if new_detections is not None and len(new_detections) > 0:
                start_id = max(self._track_id, default=0) + 1
                new_ids = list(range(start_id, start_id + len(new_detections)))
                for bbox, obj_id in zip(new_detections.xyxy, new_ids):
                    self._predictor.add_new_prompt(
                        frame_idx=self._frame_idx,
                        obj_id=int(obj_id),
                        bbox=bbox,
                    )
                self._track_id.extend(new_ids)

            tracker_ids, mask_logits = self._predictor.track(frame)

        self._frame_idx += 1
        tracker_ids = np.asarray(tracker_ids, dtype=np.int32)
        masks = (mask_logits > 0.0).cpu().numpy()
        masks = np.squeeze(masks).astype(bool)

        if masks.ndim == 2:
            masks = masks[None, ...]

        masks = np.array([
            sv.filter_segments_by_distance(mask, relative_distance=0.03, mode="edge")
            for mask in masks
        ])

        xyxy = sv.mask_to_xyxy(masks=masks)
        detections = sv.Detections(xyxy=xyxy, mask=masks, tracker_id=tracker_ids)
        return detections

    def reset(self) -> None:
        self._prompted = False
        self._track_id = []
        self._frame_idx = 0


