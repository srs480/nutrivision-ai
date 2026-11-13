export interface RawDetection {
  id: string;
  name: string;
  confidence: number;
  class_id: number;
  /** [x1, y1, x2, y2] in original image pixel coordinates */
  bbox: [number, number, number, number];
  isFoodCore: boolean;
  color: string;
}

export interface DetectionExportSchema {
  model: string;
  backend: 'webgpu' | 'wasm' | 'uninitialized';
  inferenceTimeMs: number;
  imageWidth: number;
  imageHeight: number;
  confidenceThreshold: number;
  iouThreshold: number;
  detections: Array<{
    name: string;
    confidence: number;
    class_id: number;
    bbox: [number, number, number, number];
  }>;
}

export interface LetterboxMetadata {
  scale: number;
  padX: number;
  padY: number;
  originalWidth: number;
  originalHeight: number;
  inputWidth: number;
  inputHeight: number;
}

export interface ModelProfile {
  id: string;
  name: string;
  description: string;
  localPath: string;
  fallbackUrls: string[];
  inputSize: number;
  classes: Record<number, string>;
  coreFoodClassIds: number[];
  diningContextClassIds: number[];
}

export type ModelStatus =
  | 'idle'
  | 'checking_cache'
  | 'downloading'
  | 'initializing_session'
  | 'ready'
  | 'error';