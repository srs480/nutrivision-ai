import * as ort from 'onnxruntime-web';
import {
  LetterboxMetadata,
  ModelProfile,
  RawDetection,
} from '../types/detection';
import { getCachedModelBuffer, saveCachedModelBuffer } from './modelCache';
import { preprocessImageForYolo } from '../utils/imagePipeline';
import { applyNonMaxSuppression, decodeYoloV8Output } from '../utils/nms';

// Configure WASM runtime binary delivery via CDN matching the exact installed onnxruntime-web version
ort.env.wasm.wasmPaths =
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
ort.env.wasm.simd = true;

export interface InferenceResult {
  rawUnfilteredCount: number;
  detections: RawDetection[];
  metadata: LetterboxMetadata;
  inferenceTimeMs: number;
  backend: 'webgpu' | 'wasm';
  numModelClasses: number;
}

class FoodInferenceEngine {
  private session: ort.InferenceSession | null = null;
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
   * Validates whether an ArrayBuffer starts with a valid protobuf/ONNX binary header
   * rather than an HTML 404 fallback page returned by a SPA server.
   */
  private isValidOnnxBuffer(buffer: ArrayBuffer): boolean {
    if (buffer.byteLength < 1024) return false;
    const headerBytes = new Uint8Array(buffer, 0, 15);
    const textPrefix = new TextDecoder().decode(headerBytes).trim().toLowerCase();
    if (textPrefix.startsWith('<!doctype') || textPrefix.startsWith('<html')) {
      return false;
    }
    return true;
  }

  private async fetchWithProgress(
    url: string,
    onProgress?: (progressPct: number, loadedMb: number) => void
  ): Promise<ArrayBuffer> {
    const response = await fetch(url, { mode: 'cors' });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} while fetching ${url}`);
    }

    const contentType = response.headers.get('content-type') || '';
    if (contentType.includes('text/html')) {
      throw new Error(`URL ${url} returned HTML instead of an ONNX binary.`);
    }

    const contentLength = Number(response.headers.get('content-length') || 0);
    if (!response.body) {
      const buf = await response.arrayBuffer();
      if (!this.isValidOnnxBuffer(buf)) {
        throw new Error('Downloaded file is not a valid ONNX model binary.');
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
      throw new Error('Fetched resource is not a valid ONNX protobuf binary.');
    }

    return buffer;
  }

  /**
   * Loads an ONNX model from:
   * 1. Browser IndexedDB cache (instantaneous on repeat visits)
   * 2. Local `/models/...` static path
   * 3. Configured open-source static weight fallback mirrors
   */
  public async loadModelProfile(
    profile: ModelProfile,
    onProgress?: (stage: string, pct: number) => void
  ): Promise<{ backend: 'webgpu' | 'wasm'; source: string }> {
    if (this.session && this.activeProfileId === profile.id) {
      return { backend: this.activeBackend, source: 'Memory Session' };
    }

    this.cachedRawOutput = null;
    onProgress?.('Checking local browser IndexedDB cache...', 10);

    let modelBuffer = await getCachedModelBuffer(profile.id);
    let loadedSource = 'IndexedDB Cache';

    if (!modelBuffer || !this.isValidOnnxBuffer(modelBuffer)) {
      modelBuffer = null;
      const candidateUrls = [profile.localPath, ...profile.fallbackUrls];
      let lastError: Error | null = null;

      for (const url of candidateUrls) {
        try {
          onProgress?.(`Downloading model weights (${url})...`, 20);
          modelBuffer = await this.fetchWithProgress(url, (pct, mb) => {
            onProgress?.(
              `Downloading ONNX weights (${mb} MB)...`,
              Math.max(20, Math.min(85, pct))
            );
          });
          loadedSource = url.startsWith('/') ? `Local Asset (${url})` : 'Public ONNX Mirror';
          break;
        } catch (err) {
          lastError = err instanceof Error ? err : new Error(String(err));
        }
      }

      if (!modelBuffer) {
        throw new Error(
          `Could not load ONNX model "${profile.name}". ${
            lastError?.message || ''
          } You can either run "python ml/export_model.py" to place the model in public/models/ or click "Upload Custom .onnx" in the toolbar.`
        );
      }

      onProgress?.('Caching model in IndexedDB for offline reuse...', 88);
      await saveCachedModelBuffer(profile.id, modelBuffer);
    }

    onProgress?.('Initializing ONNX Runtime Web session...', 92);
    const { session, backend } = await this.createSessionWithFallback(modelBuffer);

    this.session = session;
    this.activeProfileId = profile.id;
    this.activeBackend = backend;

    onProgress?.('Model ready', 100);
    return { backend, source: loadedSource };
  }

  /**
   * Allows a user to load a custom `.onnx` file directly from their computer disk.
   */
  public async loadCustomOnnxFile(
    file: File,
    profileId: string
  ): Promise<{ backend: 'webgpu' | 'wasm' }> {
    const buffer = await file.arrayBuffer();
    if (!this.isValidOnnxBuffer(buffer)) {
      throw new Error('Selected file is not a valid ONNX model binary.');
    }

    const { session, backend } = await this.createSessionWithFallback(buffer);
    this.session = session;
    this.activeProfileId = profileId;
    this.activeBackend = backend;
    this.cachedRawOutput = null;

    await saveCachedModelBuffer(profileId, buffer);
    return { backend };
  }

  /**
   * Attempts WebGPU execution provider first when supported by the browser,
   * gracefully falling back to WebAssembly (WASM) SIMD.
   */
  private async createSessionWithFallback(
    modelBuffer: ArrayBuffer
  ): Promise<{ session: ort.InferenceSession; backend: 'webgpu' | 'wasm' }> {
    const hasWebGpu = typeof navigator !== 'undefined' && 'gpu' in navigator;

    if (hasWebGpu) {
      try {
        const session = await ort.InferenceSession.create(modelBuffer.slice(0), {
          executionProviders: ['webgpu'],
          graphOptimizationLevel: 'all',
        });
        return { session, backend: 'webgpu' };
      } catch {
        // WebGPU failed or unsupported op; fall back to WASM
      }
    }

    const session = await ort.InferenceSession.create(modelBuffer.slice(0), {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    return { session, backend: 'wasm' };
  }

  /**
   * Executes full image preprocessing + neural network forward pass, caching the raw
   * output tensor so subsequent confidence/IoU slider adjustments happen in <2ms
   * without re-running the neural network.
   */
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
    const feeds: Record<string, ort.Tensor> = { [inputName]: inputTensor };
    const outputMap = await this.session.run(feeds);
    const inferenceTimeMs = Math.round(performance.now() - t0);

    const outputName = this.session.outputNames[0];
    const outputTensor = outputMap[outputName];

    if (!outputTensor) {
      throw new Error('Model returned an empty output tensor.');
    }

    const rawData = outputTensor.data as Float32Array;
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

  /**
   * Re-decodes the cached output tensor instantaneously when the user adjusts
   * confidence threshold, IoU threshold, or category filter mode.
   */
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
    // Only apply COCO subset filtering if the loaded model actually has 80 classes
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
