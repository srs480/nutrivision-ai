import { describe, it, expect } from 'vitest';
import {
  computeIoU,
  applyNonMaxSuppression,
  unletterboxCoords,
} from '../utils/nms';
import { RawDetection } from '../types/detection';

describe('Computer Vision Post-Processing & Coordinate Pipeline', () => {
  it('computes exact Intersection over Union (IoU) for overlapping boxes', () => {
    const boxA: [number, number, number, number] = [100, 100, 200, 200];
    const boxB: [number, number, number, number] = [100, 100, 200, 200];
    expect(computeIoU(boxA, boxB)).toBeCloseTo(1.0, 5);

    const nonOverlapping: [number, number, number, number] = [300, 300, 400, 400];
    expect(computeIoU(boxA, nonOverlapping)).toBe(0);
  });

  it('suppresses duplicate overlapping detections of the same food class while preserving distinct instances', () => {
    const detections: RawDetection[] = [
      {
        id: 'pizza-1-high',
        name: 'Pizza',
        confidence: 0.94,
        class_id: 53,
        bbox: [100, 100, 300, 300],
        isFoodCore: true,
        color: '#10B981',
      },
      {
        id: 'pizza-1-duplicate',
        name: 'Pizza',
        confidence: 0.81,
        class_id: 53,
        bbox: [105, 102, 298, 302],
        isFoodCore: true,
        color: '#10B981',
      },
      {
        id: 'pizza-2-second-slice',
        name: 'Pizza',
        confidence: 0.89,
        class_id: 53,
        bbox: [400, 100, 600, 300],
        isFoodCore: true,
        color: '#10B981',
      },
    ];

    const filtered = applyNonMaxSuppression(detections, 0.45);
    expect(filtered).toHaveLength(2);
    expect(filtered.map((d) => d.id)).toEqual([
      'pizza-1-high',
      'pizza-2-second-slice',
    ]);
  });

  it('accurately un-letterboxes 640x640 model coordinates back to original 1920x1080 image space', () => {
    const meta = {
      scale: 640 / 1920, // 0.333333
      padX: 0,
      padY: 140, // (640 - 360) / 2
      originalWidth: 1920,
      originalHeight: 1080,
      inputWidth: 640,
      inputHeight: 640,
    };

    // Center of image in 640x640 letterboxed space is (320, 320)
    const [x1, y1, x2, y2] = unletterboxCoords(320, 320, 64, 36, meta);
    expect(x1).toBeCloseTo(864, -1);
    expect(y1).toBeCloseTo(486, -1);
    expect(x2).toBeCloseTo(1056, -1);
    expect(y2).toBeCloseTo(594, -1);
  });
});