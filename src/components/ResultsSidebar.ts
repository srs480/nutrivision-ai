import React from 'react';
import {
  FileJson,
  CheckCircle,
  Target,
  Box,
  Layers,
} from 'lucide-react';
import {
  DetectionExportSchema,
  RawDetection,
} from '../types/detection';

interface ResultsSidebarProps {
  detections: RawDetection[];
  selectedDetectionId: string | null;
  onSelectDetection: (id: string | null) => void;
  modelName: string;
  backend: 'webgpu' | 'wasm';
  inferenceTimeMs: number;
  imageWidth: number;
  imageHeight: number;
  confidenceThreshold: number;
  iouThreshold: number;
}

export const ResultsSidebar: React.FC<ResultsSidebarProps> = ({
  detections,
  selectedDetectionId,
  onSelectDetection,
  modelName,
  backend,
  inferenceTimeMs,
  imageWidth,
  imageHeight,
  confidenceThreshold,
  iouThreshold,
}) => {
  // Compute category aggregation counts
  const categorySummary = React.useMemo(() => {
    const counts = new Map<
      string,
      { count: number; color: string; maxConf: number; classId: number }
    >();
    for (const det of detections) {
      const existing = counts.get(det.name);
      if (existing) {
        existing.count += 1;
        existing.maxConf = Math.max(existing.maxConf, det.confidence);
      } else {
        counts.set(det.name, {
          count: 1,
          color: det.color,
          maxConf: det.confidence,
          classId: det.class_id,
        });
      }
    }
    return Array.from(counts.entries());
  }, [detections]);

  const handleDownloadJson = () => {
    const payload: DetectionExportSchema = {
      model: modelName,
      backend,
      inferenceTimeMs,
      imageWidth,
      imageHeight,
      confidenceThreshold,
      iouThreshold,
      detections: detections.map((d) => ({
        name: d.name,
        confidence: Number(d.confidence.toFixed(2)),
        class_id: d.class_id,
        bbox: d.bbox,
      })),
    };

    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'food-detections.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <aside className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-4 sm:p-5 shadow-sm flex flex-col h-full space-y-4">
      <div className="flex items-center justify-between border-b border-slate-200 dark:border-slate-800 pb-3">
        <div>
          <h2 className="text-base font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <Target className="w-4 h-4 text-emerald-500" />
            Detection Analytics
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Structured bounding boxes & class breakdown
          </p>
        </div>

        <button
          type="button"
          disabled={detections.length === 0}
          onClick={handleDownloadJson}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium bg-emerald-500 hover:bg-emerald-600 disabled:opacity-40 text-white shadow-sm transition-colors"
        >
          <FileJson className="w-3.5 h-3.5" />
          <span>Export JSON</span>
        </button>
      </div>

      {/* KPI Summary Cards */}
      <div className="grid grid-cols-2 gap-3">
        <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-950/70 border border-slate-200/80 dark:border-slate-800">
          <div className="text-xs text-slate-500 dark:text-slate-400 flex items-center gap-1">
            <Box className="w-3.5 h-3.5 text-emerald-500" />
            Total Objects
          </div>
          <div className="text-2xl font-bold text-slate-900 dark:text-white mt-0.5 font-mono">
            {detections.length}
          </div>
        </div>

        <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-950/70 border border-slate-200/80 dark:border-slate-800">
          <div className="text-xs text-slate-500 dark:text-slate-400 flex items-center gap-1">
            <Layers className="w-3.5 h-3.5 text-sky-500" />
            Unique Classes
          </div>
          <div className="text-2xl font-bold text-slate-900 dark:text-white mt-0.5 font-mono">
            {categorySummary.length}
          </div>
        </div>
      </div>

      {/* Category Counts */}
      {categorySummary.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
            Count Per Category
          </h3>
          <div className="flex flex-wrap gap-1.5">
            {categorySummary.map(([name, info]) => (
              <span
                key={name}
                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium bg-slate-100 dark:bg-slate-800 text-slate-800 dark:text-slate-200 border border-slate-200 dark:border-slate-700"
              >
                <span
                  className="w-2.5 h-2.5 rounded-full"
                  style={{ backgroundColor: info.color }}
                />
                <span>{name}</span>
                <span className="font-mono text-xs px-1.5 py-0.2 rounded bg-slate-200 dark:bg-slate-700">
                  ×{info.count}
                </span>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Individual Detection Cards */}
      <div className="space-y-2 flex-1 overflow-y-auto max-h-[380px] pr-1">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
            Detected Instances ({detections.length})
          </h3>
          {selectedDetectionId && (
            <button
              type="button"
              onClick={() => onSelectDetection(null)}
              className="text-[11px] text-emerald-500 hover:underline"
            >
              Reset Highlight
            </button>
          )}
        </div>

        {detections.length === 0 ? (
          <div className="p-6 rounded-xl border border-dashed border-slate-200 dark:border-slate-800 text-center space-y-1">
            <p className="text-xs font-medium text-slate-600 dark:text-slate-300">
              No objects above {Math.round(confidenceThreshold * 100)}% confidence
            </p>
            <p className="text-[11px] text-slate-500 dark:text-slate-400">
              Try lowering the Confidence slider or switching the Class Filter Scope to "All Model Classes".
            </p>
          </div>
        ) : (
          detections.map((det, idx) => {
            const isSelected = selectedDetectionId === det.id;
            return (
              <button
                key={det.id}
                type="button"
                onClick={() =>
                  onSelectDetection(isSelected ? null : det.id)
                }
                className={`w-full text-left p-3 rounded-xl border transition-all flex items-center justify-between gap-3 ${
                  isSelected
                    ? 'bg-emerald-500/10 border-emerald-500 shadow-sm'
                    : 'bg-slate-50/70 dark:bg-slate-950/50 border-slate-200/80 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700'
                }`}
              >
                <div className="flex items-start gap-2.5 min-w-0">
                  <span
                    className="w-3.5 h-3.5 rounded-md shrink-0 mt-0.5"
                    style={{ backgroundColor: det.color }}
                  />
                  <div className="min-w-0">
                    <div className="text-xs font-semibold text-slate-900 dark:text-white flex items-center gap-1.5">
                      <span className="truncate">
                        #{idx + 1} {det.name}
                      </span>
                      <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-200 dark:bg-slate-800 text-slate-600 dark:text-slate-400">
                        ID:{det.class_id}
                      </span>
                    </div>
                    <div className="text-[11px] font-mono text-slate-500 dark:text-slate-400 mt-0.5">
                      bbox: [{det.bbox.join(', ')}]
                    </div>
                  </div>
                </div>

                <div className="text-right shrink-0">
                  <span className="inline-flex items-center gap-1 font-mono text-xs font-bold text-emerald-600 dark:text-emerald-400">
                    <CheckCircle className="w-3 h-3" />
                    {(det.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </button>
            );
          })
        )}
      </div>

      {/* Live JSON Output Preview */}
      <div className="pt-2 border-t border-slate-200 dark:border-slate-800">
        <details className="group">
          <summary className="cursor-pointer text-xs font-mono text-slate-500 dark:text-slate-400 hover:text-emerald-500 flex items-center justify-between">
            <span>View Structured JSON Output</span>
            <span className="text-[10px] group-open:rotate-180 transition-transform">
              ▼
            </span>
          </summary>
          <pre className="mt-2 p-3 rounded-xl bg-slate-950 text-emerald-400 font-mono text-[11px] overflow-x-auto max-h-44 border border-slate-800">
            {JSON.stringify(
              {
                detections: detections.map((d) => ({
                  name: d.name,
                  confidence: Number(d.confidence.toFixed(2)),
                  class_id: d.class_id,
                  bbox: d.bbox,
                })),
              },
              null,
              2
            )}
          </pre>
        </details>
      </div>
    </aside>
  );
};