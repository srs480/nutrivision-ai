import {
  ALLOWED_IMAGE_MIME_TYPES,
  MAX_IMAGE_FILE_SIZE_BYTES,
} from '../config/models';
import { LetterboxMetadata } from '../types/detection';

export interface ImageValidationResult {
  valid: boolean;
  error?: string;
}

export function validateImageFile(file: File): ImageValidationResult {
  if (!ALLOWED_IMAGE_MIME_TYPES.includes(file.type.toLowerCase())) {
    return {
      valid: false,
      error: `Unsupported file type "${file.type || 'unknown'}". Please upload a JPG, JPEG, PNG, or WEBP image.`,
    };
  }

  if (file.size > MAX_IMAGE_FILE_SIZE_BYTES) {
    const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
    return {
      valid: false,
      error: `Image size (${sizeMb} MB) exceeds the 15 MB maximum limit.`,
    };
  }

  if (file.size === 0) {
    return {
      valid: false,
      error: 'The selected image file is empty.',
    };
  }

  return { valid: true };
}

/**
 * Loads an HTMLImageElement from a File or Blob URL with automatic EXIF orientation handling
 * supported by modern browsers.
 */
export function loadImageElement(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      if (img.naturalWidth === 0 || img.naturalHeight === 0) {
        reject(new Error('Image decoded with zero dimensions.'));
      } else {
        resolve(img);
      }
    };
    img.onerror = () => {
      reject(new Error('Failed to decode image file. It may be corrupted.'));
    };
    img.src = src;
  });
}

/**
 * Performs YOLOv8 Letterbox Preprocessing:
 * 1. Computes aspect-ratio-preserving scale factor to fit within targetSize x targetSize (640x640).
 * 2. Centers the scaled image on a neutral gray (#727272 / RGB 114,114,114) canvas.
 * 3. Extracts RGB channels, normalizes pixel values to [0.0, 1.0], and transposes from HWC to NCHW
 *    Float32Array [1, 3, 640, 640].
 */
export function preprocessImageForYolo(
  image: HTMLImageElement,
  targetSize = 640
): {
  tensorData: Float32Array;
  metadata: LetterboxMetadata;
} {
  const originalWidth = image.naturalWidth || image.width;
  const originalHeight = image.naturalHeight || image.height;

  const scale = Math.min(
    targetSize / originalWidth,
    targetSize / originalHeight
  );
  const newUnpaddedW = Math.round(originalWidth * scale);
  const newUnpaddedH = Math.round(originalHeight * scale);

  const padX = Math.floor((targetSize - newUnpaddedW) / 2);
  const padY = Math.floor((targetSize - newUnpaddedH) / 2);

  const canvas = document.createElement('canvas');
  canvas.width = targetSize;
  canvas.height = targetSize;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });

  if (!ctx) {
    throw new Error('Failed to initialize 2D canvas context for preprocessing.');
  }

  // Standard YOLO letterbox fill value (114, 114, 114)
  ctx.fillStyle = 'rgb(114, 114, 114)';
  ctx.fillRect(0, 0, targetSize, targetSize);

  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(
    image,
    0,
    0,
    originalWidth,
    originalHeight,
    padX,
    padY,
    newUnpaddedW,
    newUnpaddedH
  );

  const imageData = ctx.getImageData(0, 0, targetSize, targetSize).data;
  const numPixels = targetSize * targetSize;
  const tensorData = new Float32Array(3 * numPixels);

  // Convert interleaved RGBA [H, W, 4] to planar RGB [1, 3, H, W] normalized to [0, 1]
  const inv255 = 1.0 / 255.0;
  for (let i = 0; i < numPixels; i++) {
    const rgbaOffset = i * 4;
    tensorData[i] = imageData[rgbaOffset] * inv255; // R
    tensorData[numPixels + i] = imageData[rgbaOffset + 1] * inv255; // G
    tensorData[2 * numPixels + i] = imageData[rgbaOffset + 2] * inv255; // B
  }

  return {
    tensorData,
    metadata: {
      scale,
      padX,
      padY,
      originalWidth,
      originalHeight,
      inputWidth: targetSize,
      inputHeight: targetSize,
    },
  };
}

/**
 * Generates a realistic synthetic multi-food test plate on an HTML5 Canvas
 * or loads a real Wikimedia Commons multi-food image so users can immediately
 * test detection without searching for an image file.
 */
export const SAMPLE_TEST_IMAGES = [
  {
    name: 'Pizza, Salad & Drinks Spread',
    description: 'Multiple food items & tableware on a dining table',
    url: 'https://images.unsplash.com/photo-1513104890138-7c749659a591?auto=format&fit=crop&w=1000&q=80',
  },
  {
    name: 'Burger, Sandwich & Sides',
    description: 'High-contrast multi-item fast food platter',
    url: 'https://images.unsplash.com/photo-1568901346375-23c9450c58cd?auto=format&fit=crop&w=1000&q=80',
  },
  {
    name: 'Fresh Fruits & Vegetables Basket',
    description: 'Apples, Bananas, Oranges & Broccoli',
    url: 'https://images.unsplash.com/photo-1610832958506-aa56368176cf?auto=format&fit=crop&w=1000&q=80',
  },
];