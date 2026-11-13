import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Header } from './components/Header';
import { ModelControlBar } from './components/ModelControlBar';
import { ImageUploader } from './components/ImageUploader';
import { DetectionCanvas } from './components/DetectionCanvas';
import { ResultsSidebar } from './components/ResultsSidebar';
import { MODEL_PROFILES } from './config/models';
import {
  ModelProfile,
  ModelStatus,
  RawDetection,
} from './types/detection';
import { inferenceEngine } from './services/inferenceEngine';
import {
  loadImageElement,
  validateImageFile,
} from './utils/imagePipeline';

export const App: React.FC = () => {
  const [darkMode, setDarkMode] = useState(true);
  const [selectedProfile, setSelectedProfile] = useState<ModelProfile>(
    MODEL_PROFILES[0]
  );
  const [customModelName, setCustomModelName] = useState<string | null>(null);

  const [modelStatus, setModelStatus] = useState<ModelStatus>('idle');
  const [loadingStageText, setLoadingStageText] = useState<string>('');
  const [loadingProgressPct, setLoadingProgressPct] = useState<number>(0);
  const [backend, setBackend] = useState<'webgpu' | 'wasm'>('wasm');
  const [modelSource, setModelSource] = useState<string>('');
  const [errorBanner, setErrorBanner] = useState<string | null>(null);

  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.28);
  const [iouThreshold, setIouThreshold] = useState<number>(0.45);
  const [filterMode, setFilterMode] = useState<
    'food_only' | 'food_and_dining' | 'all_classes'
  >('food_and_dining');

  const [imageElement, setImageElement] = useState<HTMLImageElement | null>(
    null
  );
  const [imageName, setImageName] = useState<string>('');
  const [detections, setDetections] = useState<RawDetection[]>([]);
  const [selectedDetectionId, setSelectedDetectionId] = useState<string | null>(
    null
  );
  const [inferenceTimeMs, setInferenceTimeMs] = useState<number>(0);
  const [isProcessingImage, setIsProcessingImage] = useState<boolean>(false);

  const activeObjectUrlRef = useRef<string | null>(null);

  // Sync Dark Mode class on <html>
  useEffect(() => {
    const root = document.documentElement;
    if (darkMode) {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  }, [darkMode]);

  // Clean up temporary Blob URLs on unmount
  useEffect(() => {
    return () => {
      if (activeObjectUrlRef.current) {
        URL.revokeObjectURL(activeObjectUrlRef.current);
      }
    };
  }, []);

  // Initialize or switch ONNX model profile
  const initModel = useCallback(async (profile: ModelProfile) => {
    setModelStatus('downloading');
    setErrorBanner(null);
    setCustomModelName(null);

    try {
      const result = await inferenceEngine.loadModelProfile(
        profile,
        (stage, pct) => {
          setLoadingStageText(stage);
          setLoadingProgressPct(pct);
        }
      );
      setBackend(result.backend);
      setModelSource(result.source);
      setModelStatus('ready');
    } catch (err) {
      setModelStatus('error');
      setErrorBanner(
        err instanceof Error
          ? err.message
          : 'Failed to load ONNX model weights.'
      );
    }
  }, []);

  useEffect(() => {
    initModel(selectedProfile);
  }, [selectedProfile, initModel]);

  // Run inference whenever a new image is loaded or the model changes
  const executeInferenceOnImage = useCallback(
    async (img: HTMLImageElement) => {
      if (!inferenceEngine.isReady()) return;
      setIsProcessingImage(true);
      setErrorBanner(null);

      try {
        const res = await inferenceEngine.runInference(
          img,
          selectedProfile,
          confidenceThreshold,
          iouThreshold,
          filterMode
        );
        setDetections(res.detections);
        setInferenceTimeMs(res.inferenceTimeMs);
        setBackend(res.backend);
        setSelectedDetectionId(null);
      } catch (err) {
        setErrorBanner(
          err instanceof Error
            ? err.message
            : 'An error occurred during ONNX tensor inference.'
        );
      } finally {
        setIsProcessingImage(false);
      }
    },
    [selectedProfile, confidenceThreshold, iouThreshold, filterMode]
  );

  // Instantaneous re-decoding when sliders or filter scope change
  useEffect(() => {
    if (!imageElement || !inferenceEngine.isReady()) return;
    try {
      const res = inferenceEngine.decodeCachedOutput(
        selectedProfile,
        confidenceThreshold,
        iouThreshold,
        filterMode
      );
      setDetections(res.detections);
    } catch {
      // If no cached output exists yet for the current model, run full inference
      executeInferenceOnImage(imageElement);
    }
  }, [
    confidenceThreshold,
    iouThreshold,
    filterMode,
    selectedProfile,
    imageElement,
    executeInferenceOnImage,
  ]);

  const handleSelectFile = async (file: File) => {
    const validation = validateImageFile(file);
    if (!validation.valid) {
      setErrorBanner(validation.error || 'Invalid image file.');
      return;
    }

    setErrorBanner(null);
    setIsProcessingImage(true);

    try {
      if (activeObjectUrlRef.current) {
        URL.revokeObjectURL(activeObjectUrlRef.current);
      }
      const objectUrl = URL.createObjectURL(file);
      activeObjectUrlRef.current = objectUrl;

      const img = await loadImageElement(objectUrl);
      setImageElement(img);
      setImageName(file.name);
      await executeInferenceOnImage(img);
    } catch (err) {
      setErrorBanner(
        err instanceof Error ? err.message : 'Failed to decode uploaded image.'
      );
    } finally {
      setIsProcessingImage(false);
    }
  };

  const handleSelectSampleUrl = async (url: string, name: string) => {
    setErrorBanner(null);
    setIsProcessingImage(true);

    try {
      const img = await loadImageElement(url);
      setImageElement(img);
      setImageName(`${name}.jpg`);
      await executeInferenceOnImage(img);
    } catch (err) {
      setErrorBanner(
        err instanceof Error
          ? err.message
          : 'Failed to load sample image. Please upload a local image file.'
      );
    } finally {
      setIsProcessingImage(false);
    }
  };

  const handleUploadCustomOnnx = async (file: File) => {
    setModelStatus('initializing_session');
    setLoadingStageText(`Loading custom model ${file.name}...`);
    setLoadingProgressPct(60);
    setErrorBanner(null);

    try {
      const res = await inferenceEngine.loadCustomOnnxFile(
        file,
        selectedProfile.id
      );
      setBackend(res.backend);
      setCustomModelName(file.name);
      setModelSource(`Custom File (${file.name})`);
      setModelStatus('ready');

      if (imageElement) {
        await executeInferenceOnImage(imageElement);
      }
    } catch (err) {
      setModelStatus('error');
      setErrorBanner(
        err instanceof Error ? err.message : 'Failed to load custom ONNX file.'
      );
    }
  };

  const handleClearWorkspace = () => {
    if (activeObjectUrlRef.current) {
      URL.revokeObjectURL(activeObjectUrlRef.current);
      activeObjectUrlRef.current = null;
    }
    setImageElement(null);
    setImageName('');
    setDetections([]);
    setSelectedDetectionId(null);
    setErrorBanner(null);
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-100 dark:bg-slate-950 text-slate-900 dark:text-slate-100">
      <Header
        darkMode={darkMode}
        onToggleDarkMode={() => setDarkMode((d) => !d)}
        modelStatus={modelStatus}
        backend={backend}
        modelSource={modelSource}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        <ModelControlBar
          selectedProfile={selectedProfile}
          onSelectProfile={setSelectedProfile}
          onUploadCustomOnnx={handleUploadCustomOnnx}
          confidenceThreshold={confidenceThreshold}
          onConfidenceChange={setConfidenceThreshold}
          iouThreshold={iouThreshold}
          onIouChange={setIouThreshold}
          filterMode={filterMode}
          onFilterModeChange={setFilterMode}
          modelStatus={modelStatus}
          loadingStageText={loadingStageText}
          loadingProgressPct={loadingProgressPct}
          onReloadModel={() => initModel(selectedProfile)}
          customModelName={customModelName}
        />

        {!imageElement ? (
          <ImageUploader
            onSelectFile={handleSelectFile}
            onSelectSampleUrl={handleSelectSampleUrl}
            error={errorBanner}
            isProcessing={isProcessingImage || modelStatus !== 'ready'}
          />
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            <div className="lg:col-span-8 space-y-4">
              {errorBanner && (
                <div className="p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-600 dark:text-rose-300 text-xs">
                  {errorBanner}
                </div>
              )}
              <DetectionCanvas
                imageElement={imageElement}
                imageName={imageName}
                detections={detections}
                selectedDetectionId={selectedDetectionId}
                onSelectDetection={setSelectedDetectionId}
                onReplaceImage={handleSelectFile}
                onClearImage={handleClearWorkspace}
                inferenceTimeMs={inferenceTimeMs}
              />
            </div>

            <div className="lg:col-span-4">
              <ResultsSidebar
                detections={detections}
                selectedDetectionId={selectedDetectionId}
                onSelectDetection={setSelectedDetectionId}
                modelName={customModelName || selectedProfile.name}
                backend={backend}
                inferenceTimeMs={inferenceTimeMs}
                imageWidth={imageElement.naturalWidth}
                imageHeight={imageElement.naturalHeight}
                confidenceThreshold={confidenceThreshold}
                iouThreshold={iouThreshold}
              />
            </div>
          </div>
        )}
      </main>

      <footer className="border-t border-slate-200 dark:border-slate-900 py-4 px-4 text-center text-xs text-slate-500 dark:text-slate-500">
        NutriVision AI — Powered by YOLOv8 & ONNX Runtime Web (WebGPU / WASM). Zero External AI APIs.
      </footer>
    </div>
  );
};

export default App;
