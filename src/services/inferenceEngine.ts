import type * as OrtType from 'onnxruntime-web';
import {
  LetterboxMetadata,
  ModelProfile,
  RawDetection,
} from '../types/detection';
import {
  clearModelCache,
  getCachedModelBuffer,
  saveCachedModelBuffer,
} from './modelCache';
import { preprocessImageForYolo } from '../utils/imagePipeline';
import { applyNonMaxSuppression, decodeYoloV8Output } from '../utils/nms';

declare global {
  interface Window {
    ort?: typeof OrtType;
  }
}

function getOrtRuntime(): typeof OrtType {
  const ort = window.ort;
  if (!ort) {
    throw new Error(
      'ONNX Runtime Web global script not loaded. Please check index.html.'
    );
  }
  ort.env.wasm.wasmPaths =
    'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  return ort;
}

export interface InferenceResult {
  rawUnfilteredCount: number;
  detections: RawDetection[];
  metadata: LetterboxMetadata;
  inferenceTimeMs: number;
  backend: 'webgpu' | 'wasm';
  numModelClasses: number;
}

class FoodInferenceEngine {
  private session: OrtType.InferenceSession | null = null;
  private activeProfileId: string | null = null;
  private activeBackend: 'webgpu' | 'wasm' = 'wasm';
  private cachedRawOutput: {
    data: Float32Array;
    dims: readonly number[];
    metadata: LetterboxMetadata;
    inferenceTimeMs: number;
  } | null = null;

  public getBackend(): 'webgpu' | 'wasm' {
    return this.activeBackend;
  }

  public isReady(): boolean {
    return this.session !== null;
  }

  /**
   * Checks that the downloaded binary is at least 4 MB (real YOLOv8n ONNX is ~12.2 MB)
   * and is not an HTML/JSON/Git-LFS text file.
   */
  private isValidOnnxBuffer(buffer: ArrayBuffer): boolean {
    if (!buffer || buffer.byteLength < 4 * 1024 * 1024) {
      return false;
    }
    const headerBytes = new Uint8Array(buffer, 0, 32);
    const textPrefix = new TextDecoder()
      .decode(headerBytes)
      .trim()
      .toLowerCase();
    if (
      textPrefix.startsWith('<!doctype') ||
      textPrefix.startsWith('<html') ||
      textPrefix.startsWith('{') ||
      textPrefix.startsWith('version https://git-lfs')
    ) {
      return false;
    }
    return true;
  }

  private async fetchWithProgress(
    url: string,
    onProgress?: (progressPct: number, loadedMb: number) => void
  ): Promise<ArrayBuffer> {
    const response = await fetch(url, { mode: 'cors', cache: 'force-cache' });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} while fetching ${url}`);
    }

    const contentLength = Number(response.headers.get('content-length') || 0);
    if (!response.body) {
      const buf = await response.arrayBuffer();
      if (!this.isValidOnnxBuffer(buf)) {
        throw new Error(`Invalid ONNX binary from ${url}`);
      }
      return buf;
    }

    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    let receivedLength = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        chunks.push(value);
        receivedLength += value.length;
        if (onProgress) {
          const pct =
            contentLength > 0
              ? Math.min(99, Math.round((receivedLength / contentLength) * 100))
              : 50;
          onProgress(pct, Number((receivedLength / (1024 * 1024)).toFixed(2)));
        }
      }
    }

    const combined = new Uint8Array(receivedLength);
    let position = 0;
    for (const chunk of chunks) {
      combined.set(chunk, position);
      position += chunk.length;
    }

    const buffer = combined.buffer;
    if (!this.isValidOnnxBuffer(buffer)) {
      throw new Error(`Resource from ${url} is smaller than 4MB or not a valid ONNX file.`);
    }

    return buffer;
  }

  public async loadModelProfile(
    profile: ModelProfile,
    onProgress?: (stage: string, pct: number) => void
  ): Promise<{ backend: 'webgpu' | 'wasm'; source: string }> {
    if (this.session && this.activeProfileId === profile.id) {
      return { backend: this.activeBackend, source: 'Memory Session' };
    }

    this.cachedRawOutput = null;
    onProgress?.('Checking local browser cache...', 10);

    let modelBuffer = await getCachedModelBuffer(profile.id);
    let loadedSource = 'IndexedDB Cache';

    if (modelBuffer && this.isValidOnnxBuffer(modelBuffer)) {
      try {
        onProgress?.('Initializing cached ONNX session...', 90);
        const { session, backend } = await this.createSessionSafely(modelBuffer);
        this.session = session;
        this.activeProfileId = profile.id;
        this.activeBackend = backend;
        onProgress?.('Model ready', 100);
        return { backend, source: loadedSource };
      } catch {
        // Cached buffer was corrupted; clear cache and re-download
        await clearModelCache();
        modelBuffer = null;
      }
    }

    const candidateUrls = [profile.localPath, ...profile.fallbackUrls];
    let lastError: Error | null = null;

    for (const url of candidateUrls) {
      try {
        onProgress?.(`Downloading ONNX model weights...`, 20);
        modelBuffer = await this.fetchWithProgress(url, (pct, mb) => {
          onProgress?.(
            `Downloading ONNX weights (${mb} MB)...`,
            Math.max(20, Math.min(85, pct))
          );
        });
        loadedSource = url.startsWith('/')
          ? `Local Asset (${url})`
          : 'Public ONNX Mirror';
        break;
      } catch (err) {
        lastError = err instanceof Error ? err : new Error(String(err));
      }
    }

    if (!modelBuffer) {
      throw new Error(
        `Could not download ONNX model "${profile.name}". ${
          lastError?.message || ''
        }`
      );
    }

    onProgress?.('Initializing ONNX Runtime Web WASM engine...', 92);
    const { session, backend } = await this.createSessionSafely(modelBuffer);

    onProgress?.('Caching verified model in browser...', 97);
    await saveCachedModelBuffer(profile.id, modelBuffer);

    this.session = session;
    this.activeProfileId = profile.id;
    this.activeBackend = backend;

    onProgress?.('Model ready', 100);
    return { backend, source: loadedSource };
  }

  public async loadCustomOnnxFile(
    file: File,
    profileId: string
  ): Promise<{ backend: 'webgpu' | 'wasm' }> {
    const buffer = await file.arrayBuffer();
    if (buffer.byteLength < 100 * 1024) {
      throw new Error('Selected file is too small to be a valid YOLO ONNX model.');
    }

    const { session, backend } = await this.createSessionSafely(buffer);
    this.session = session;
    this.activeProfileId = profileId;
    this.activeBackend = backend;
    this.cachedRawOutput = null;

    await saveCachedModelBuffer(profileId, buffer);
    return { backend };
  }

  private async createSessionSafely(
    modelBuffer: ArrayBuffer
  ): Promise<{ session: OrtType.InferenceSession; backend: 'webgpu' | 'wasm' }> {
    const ort = getOrtRuntime();
    const modelBytes = new Uint8Array(modelBuffer.slice(0));

    const session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'basic',
    });

    return { session, backend: 'wasm' };
  }

  public async runInference(
    image: HTMLImageElement,
    profile: ModelProfile,
    confidenceThreshold: number,
    iouThreshold: number,
    filterMode: 'food_only' | 'food_and_dining' | 'all_classes'
  ): Promise<InferenceResult> {
    if (!this.session) {
      throw new Error('ONNX InferenceSession is not initialized.');
    }

    const ort = getOrtRuntime();
    const { tensorData, metadata } = preprocessImageForYolo(
      image,
      profile.inputSize
    );

    const inputName = this.session.inputNames[0] || 'images';
    const inputTensor = new ort.Tensor('float32', tensorData, [
      1,
      3,
      profile.inputSize,
      profile.inputSize,
    ]);

    const t0 = performance.now();
    const feeds: Record<string, OrtType.Tensor> = { [inputName]: inputTensor };
    const outputMap = await this.session.run(feeds);
    const inferenceTimeMs = Math.round(performance.now() - t0);

    const outputName = this.session.outputNames[0];
    const outputTensor = outputMap[outputName];

    if (!outputTensor) {
      throw new Error('Model returned an empty output tensor.');
    }

    const rawData =
      outputTensor.data instanceof Float32Array
        ? outputTensor.data
        : new Float32Array(outputTensor.data as ArrayLike<number>);

    const dims = outputTensor.dims;

    this.cachedRawOutput = {
      data: rawData,
      dims,
      metadata,
      inferenceTimeMs,
    };

    return this.decodeCachedOutput(
      profile,
      confidenceThreshold,
      iouThreshold,
      filterMode
    );
  }

  public decodeCachedOutput(
    profile: ModelProfile,
    confidenceThreshold: number,
    iouThreshold: number,
    filterMode: 'food_only' | 'food_and_dining' | 'all_classes'
  ): InferenceResult {
    if (!this.cachedRawOutput) {
      throw new Error('No cached inference output available.');
    }

    const { data, dims, metadata, inferenceTimeMs } = this.cachedRawOutput;
    const numModelClasses = dims[1] - 4;

    let allowedClassIds: Set<number> | null = null;
    if (numModelClasses === 80) {
      if (filterMode === 'food_only') {
        allowedClassIds = new Set(profile.coreFoodClassIds);
      } else if (filterMode === 'food_and_dining') {
        allowedClassIds = new Set([
          ...profile.coreFoodClassIds,
          ...profile.diningContextClassIds,
        ]);
      }
    }

    const candidates = decodeYoloV8Output(
      data,
      dims,
      metadata,
      profile.classes,
      profile.coreFoodClassIds,
      allowedClassIds,
      confidenceThreshold
    );

    const nmsDetections = applyNonMaxSuppression(candidates, iouThreshold);

    return {
      rawUnfilteredCount: candidates.length,
      detections: nmsDetections,
      metadata,
      inferenceTimeMs,
      backend: this.activeBackend,
      numModelClasses,
    };
  }
}

export const inferenceEngine = new FoodInferenceEngine();
