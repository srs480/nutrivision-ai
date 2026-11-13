import React, { useRef } from 'react';
import {
  Sliders,
  Upload,
  Info,
  Layers,
  RefreshCw,
} from 'lucide-react';
import { MODEL_PROFILES } from '../config/models';
import { ModelProfile, ModelStatus } from '../types/detection';

interface ModelControlBarProps {
  selectedProfile: ModelProfile;
  onSelectProfile: (profile: ModelProfile) => void;
  onUploadCustomOnnx: (file: File) => void;
  confidenceThreshold: number;
  onConfidenceChange: (value: number) => void;
  iouThreshold: number;
  onIouChange: (value: number) => void;
  filterMode: 'food_only' | 'food_and_dining' | 'all_classes';
  onFilterModeChange: (mode: 'food_only' | 'food_and_dining' | 'all_classes') => void;
  modelStatus: ModelStatus;
  loadingStageText: string;
  loadingProgressPct: number;
  onReloadModel: () => void;
  customModelName: string | null;
}

export const ModelControlBar: React.FC<ModelControlBarProps> = ({
  selectedProfile,
  onSelectProfile,
  onUploadCustomOnnx,
  confidenceThreshold,
  onConfidenceChange,
  iouThreshold,
  onIouChange,
  filterMode,
  onFilterModeChange,
  modelStatus,
  loadingStageText,
  loadingProgressPct,
  onReloadModel,
  customModelName,
}) => {
  const onnxInputRef = useRef<HTMLInputElement | null>(null);
  const [showSupportedClasses, setShowSupportedClasses] = React.useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onUploadCustomOnnx(file);
      e.target.value = '';
    }
  };

  return (
    <section className="bg-white dark:bg-slate-900/90 border border-slate-200 dark:border-slate-800 rounded-2xl p-4 sm:p-5 shadow-sm space-y-4">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-center">
        {/* Model Profile Selector */}
        <div className="lg:col-span-5 space-y-1.5">
          <div className="flex items-center justify-between">
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-emerald-500" />
              Active Computer Vision Model
            </label>
            <button
              type="button"
              onClick={() => setShowSupportedClasses((prev) => !prev)}
              className="text-xs text-emerald-600 dark:text-emerald-400 hover:underline flex items-center gap-1"
            >
              <Info className="w-3.5 h-3.5" />
              {showSupportedClasses ? 'Hide Model Classes' : 'Inspect Supported Classes'}
            </button>
          </div>
          <div className="flex items-center gap-2">
            <select
              value={selectedProfile.id}
              onChange={(e) => {
                const found = MODEL_PROFILES.find((p) => p.id === e.target.value);
                if (found) onSelectProfile(found);
              }}
              className="w-full rounded-xl bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 px-3 py-2 text-sm text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
            >
              {MODEL_PROFILES.map((profile) => (
                <option key={profile.id} value={profile.id}>
                  {profile.name}
                </option>
              ))}
            </select>

            <input
              ref={onnxInputRef}
              type="file"
              accept=".onnx"
              onChange={handleFileChange}
              className="hidden"
            />
            <button
              type="button"
              onClick={() => onnxInputRef.current?.click()}
              title="Load a custom YOLOv8 .onnx weight file directly from your disk"
              className="shrink-0 inline-flex items-center gap-1.5 px-3 py-2 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-xs font-medium text-slate-700 dark:text-slate-200 transition-colors"
            >
              <Upload className="w-3.5 h-3.5 text-emerald-500" />
              <span>Load .ONNX</span>
            </button>

            <button
              type="button"
              onClick={onReloadModel}
              title="Reload model weights"
              className="p-2 rounded-xl border border-slate-200 dark:border-slate-800 text-slate-500 hover:text-slate-800 dark:hover:text-slate-200"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
          {customModelName && (
            <p className="text-xs text-emerald-600 dark:text-emerald-400 font-mono">
              Custom File Active: {customModelName}
            </p>
          )}
        </div>

        {/* Category Scope Filter */}
        <div className="lg:col-span-3 space-y-1.5">
          <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 block">
            Class Filter Scope
          </label>
          <select
            value={filterMode}
            onChange={(e) =>
              onFilterModeChange(
                e.target.value as 'food_only' | 'food_and_dining' | 'all_classes'
              )
            }
            className="w-full rounded-xl bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 px-3 py-2 text-sm text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
          >
            <option value="food_and_dining">
              Food + Bowls/Dishes (Recommended)
            </option>
            <option value="food_only">Strict Core Food Classes Only</option>
            <option value="all_classes">All Model Classes (Unfiltered)</option>
          </select>
        </div>

        {/* Confidence & NMS IoU Threshold Sliders */}
        <div className="lg:col-span-4 grid grid-cols-2 gap-3">
          <div className="bg-slate-50 dark:bg-slate-950/70 p-2.5 rounded-xl border border-slate-200/80 dark:border-slate-800">
            <div className="flex items-center justify-between text-xs mb-1">
              <span className="font-medium text-slate-600 dark:text-slate-400 flex items-center gap-1">
                <Sliders className="w-3 h-3 text-emerald-500" />
                Confidence
              </span>
              <span className="font-mono font-semibold text-emerald-600 dark:text-emerald-400">
                {Math.round(confidenceThreshold * 100)}%
              </span>
            </div>
            <input
              type="range"
              min={0.05}
              max={0.95}
              step={0.01}
              value={confidenceThreshold}
              onChange={(e) => onConfidenceChange( parseFloat(e.target.value) )}
              className="w-full accent-emerald-500 cursor-pointer h-1.5 bg-slate-200 dark:bg-slate-800 rounded-lg"
            />
          </div>

          <div className="bg-slate-50 dark:bg-slate-950/70 p-2.5 rounded-xl border border-slate-200/80 dark:border-slate-800">
            <div className="flex items-center justify-between text-xs mb-1">
              <span className="font-medium text-slate-600 dark:text-slate-400">
                NMS IoU Limit
              </span>
              <span className="font-mono font-semibold text-sky-600 dark:text-sky-400">
                {Math.round(iouThreshold * 100)}%
              </span>
            </div>
            <input
              type="range"
              min={0.15}
              max={0.9}
              step={0.05}
              value={iouThreshold}
              onChange={(e) => onIouChange(parseFloat(e.target.value))}
              className="w-full accent-sky-500 cursor-pointer h-1.5 bg-slate-200 dark:bg-slate-800 rounded-lg"
            />
          </div>
        </div>
      </div>

      {/* Model Download Progress Bar */}
      {(modelStatus === 'downloading' ||
        modelStatus === 'checking_cache' ||
        modelStatus === 'initializing_session') && (
        <div className="pt-1 space-y-1.5">
          <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
            <span>{loadingStageText}</span>
            <span className="font-mono">{loadingProgressPct}%</span>
          </div>
          <div className="w-full h-2 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-emerald-500 to-teal-400 transition-all duration-300"
              style={{ width: `${loadingProgressPct}%` }}
            />
          </div>
        </div>
      )}

      {/* Collapsible Transparent Class Documentation */}
      {showSupportedClasses && (
        <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 text-xs space-y-2">
          <p className="text-slate-600 dark:text-slate-300 leading-relaxed">
            <strong>Model Capabilities & Honesty Notice:</strong> {selectedProfile.description}
          </p>
          <div className="flex flex-wrap gap-1.5 pt-1">
            {[
              ...selectedProfile.coreFoodClassIds,
              ...selectedProfile.diningContextClassIds,
            ].map((id) => (
              <span
                key={id}
                className="px-2 py-0.5 rounded-md font-mono text-[11px] bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 border border-emerald-500/20"
              >
                [{id}] {selectedProfile.classes[id]}
              </span>
            ))}
          </div>
        </div>
      )}
    </section>
  );
};
