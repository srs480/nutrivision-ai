import { RawDetection, LetterboxMetadata } from '../types/detection';
import { getClassColor } from './colors';

/**
 * Calculates Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2].
 */
export function computeIoU(
  boxA: [number, number, number, number],
  boxB: [number, number, number, number]
): number {
  const xA = Math.max(boxA[0], boxB[0]);
  const yA = Math.max(boxA[1], boxB[1]);
  const xB = Math.min(boxA[2], boxB[2]);
  const yB = Math.min(boxA[3], boxB[3]);

  const interWidth = Math.max(0, xB - xA);
  const interHeight = Math.max(0, yB - yA);
  const interArea = interWidth * interHeight;

  if (interArea <= 0) return 0;

  const boxAArea = Math.max(0, boxA[2] - boxA[0]) * Math.max(0, boxA[3] - boxA[1]);
  const boxBArea = Math.max(0, boxB[2] - boxB[0]) * Math.max(0, boxB[3] - boxB[1]);
  const unionArea = boxAArea + boxBArea - interArea;

  return unionArea > 0 ? interArea / unionArea : 0;
}

/**
 * Class-aware Non-Maximum Suppression (NMS) with fallback protection for strongly overlapping
 * duplicate boxes.
 */
export function applyNonMaxSuppression(
  detections: RawDetection[],
  iouThreshold: number,
  maxDetections = 100
): RawDetection[] {
  const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
  const selected: RawDetection[] = [];

  for (const candidate of sorted) {
    if (selected.length >= maxDetections) break;

    let shouldSuppress = false;
    for (const kept of selected) {
      // Suppress within same class at standard IoU threshold, or across classes if nearly identical (>0.88 IoU)
      const iou = computeIoU(candidate.bbox, kept.bbox);
      if (
        (candidate.class_id === kept.class_id && iou >= iouThreshold) ||
        iou >= 0.88
      ) {
        shouldSuppress = true;
        break;
      }
    }

    if (!shouldSuppress) {
      selected.push(candidate);
    }
  }

  return selected;
}

/**
 * Maps letterboxed [cx, cy, w, h] coordinates in 640x640 model space back to
 * original image pixel coordinates [x1, y1, x2, y2], clamped to image bounds.
 */
export function unletterboxCoords(
  cx: number,
  cy: number,
  w: number,
  h: number,
  meta: LetterboxMetadata
): [number, number, number, number] {
  const x1Model = cx - w / 2;
  const y1Model = cy - h / 2;
  const x2Model = cx + w / 2;
  const y2Model = cy + h / 2;

  const x1Orig = (x1Model - meta.padX) / meta.scale;
  const y1Orig = (y1Model - meta.padY) / meta.scale;
  const x2Orig = (x2Model - meta.padX) / meta.scale;
  const y2Orig = (y2Model - meta.padY) / meta.scale;

  const clampedX1 = Math.round(Math.max(0, Math.min(meta.originalWidth, x1Orig)));
  const clampedY1 = Math.round(Math.max(0, Math.min(meta.originalHeight, y1Orig)));
  const clampedX2 = Math.round(Math.max(0, Math.min(meta.originalWidth, x2Orig)));
  const clampedY2 = Math.round(Math.max(0, Math.min(meta.originalHeight, y2Orig)));

  return [clampedX1, clampedY1, clampedX2, clampedY2];
}

/**
 * Decodes raw YOLOv8 output tensor [1, 4 + numClasses, numAnchors] (typically [1, 84, 8400])
 * into filtered and un-letterboxed detections.
 */
export function decodeYoloV8Output(
  outputData: Float32Array,
  dimensions: readonly number[],
  meta: LetterboxMetadata,
  classMap: Record<number, string>,
  coreFoodClassIds: number[],
  allowedClassIds: Set<number> | null,
  minConfidence: number
): RawDetection[] {
  // YOLOv8 default output shape: [1, 4 + numClasses, 8400]
  const numChannels = dimensions[1];
  const numAnchors = dimensions[2];
  const numClasses = numChannels - 4;

  const rawDetections: RawDetection[] = [];
  const coreSet = new Set(coreFoodClassIds);

  for (let anchorIdx = 0; anchorIdx < numAnchors; anchorIdx++) {
    let bestClassId = -1;
    let maxScore = 0;

    for (let classId = 0; classId < numClasses; classId++) {
      if (allowedClassIds !== null && !allowedClassIds.has(classId)) {
        continue;
      }
      const score = outputData[(4 + classId) * numAnchors + anchorIdx];
      if (score > maxScore) {
        maxScore = score;
        bestClassId = classId;
      }
    }

    if (bestClassId === -1 || maxScore < minConfidence) {
      continue;
    }

    const cx = outputData[0 * numAnchors + anchorIdx];
    const cy = outputData[1 * numAnchors + anchorIdx];
    const w = outputData[2 * numAnchors + anchorIdx];
    const h = outputData[3 * numAnchors + anchorIdx];

    const bbox = unletterboxCoords(cx, cy, w, h, meta);
    const boxWidth = bbox[2] - bbox[0];
    const boxHeight = bbox[3] - bbox[1];

    if (boxWidth < 4 || boxHeight < 4) {
      continue;
    }

    const className = classMap[bestClassId] ?? `Class #${bestClassId}`;

    rawDetections.push({
      id: `${bestClassId}-${anchorIdx}`,
      name: className,
      confidence: Number(maxScore.toFixed(4)),
      class_id: bestClassId,
      bbox,
      isFoodCore: coreSet.has(bestClassId),
      color: getClassColor(bestClassId),
    });
  }

  return rawDetections;
}