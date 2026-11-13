import React, { useEffect, useRef, useState } from 'react';
import {
  Eye,
  EyeOff,
  Download,
  RefreshCw,
  Trash2,
  Columns,
  Maximize2,
} from 'lucide-react';
import { RawDetection } from '../types/detection';
import { hexToRgba } from '../utils/colors';

interface DetectionCanvasProps {
  imageElement: HTMLImageElement;
  imageName: string;
  detections: RawDetection[];
  selectedDetectionId: string | null;
  onSelectDetection: (id: string | null) => void;
  onReplaceImage: (file: File) => void;
  onClearImage: () => void;
  inferenceTimeMs: number;
}

export const DetectionCanvas: React.FC<DetectionCanvasProps> = ({
  imageElement,
  imageName,
  detections,
  selectedDetectionId,
  onSelectDetection,
  onReplaceImage,
  onClearImage,
  inferenceTimeMs,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const replaceInputRef = useRef<HTMLInputElement | null>(null);
  const [showOverlay, setShowOverlay] = useState(true);
  const [sideBySideView, setSideBySideView] = useState(false);

  // Render image and bounding boxes onto the primary canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageElement) return;

    const width = imageElement.naturalWidth || imageElement.width;
    const height = imageElement.naturalHeight || imageElement.height;
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(imageElement, 0, 0, width, height);

    if (!showOverlay) return;

    // Scale line width & font size proportionally to image resolution
    const scaleFactor = Math.max(1, Math.min(width, height) / 650);
    const boxLineWidth = Math.round(2.5 * scaleFactor);
    const fontSize = Math.round(14 * scaleFactor);
    ctx.font = `600 ${fontSize}px Inter, system-ui, sans-serif`;

    detections.forEach((det) => {
      const [x1, y1, x2, y2] = det.bbox;
      const boxW = x2 - x1;
      const boxH = y2 - y1;
      const isSelected = selectedDetectionId === det.id;
      const isDimmed = selectedDetectionId !== null && !isSelected;

      ctx.save();

      // Semi-transparent fill
      ctx.fillStyle = hexToRgba(
        det.color,
        isSelected ? 0.28 : isDimmed ? 0.05 : 0.14
      );
      ctx.fillRect(x1, y1, boxW, boxH);

      // Bounding box stroke
      ctx.strokeStyle = det.color;
      ctx.lineWidth = isSelected ? boxLineWidth * 1.6 : boxLineWidth;
      ctx.globalAlpha = isDimmed ? 0.35 : 1.0;
      ctx.strokeRect(x1, y1, boxW, boxH);

      // Label Pill
      const labelText = `${det.name} ${(det.confidence * 100).toFixed(0)}%`;
      const textMetrics = ctx.measureText(labelText);
      const padX = Math.round(7 * scaleFactor);
      const padY = Math.round(4 * scaleFactor);
      const pillW = textMetrics.width + padX * 2;
      const pillH = fontSize + padY * 2;

      const pillX = Math.max(0, Math.min(width - pillW, x1));
      const pillY = y1 - pillH >= 0 ? y1 - pillH : y1;

      ctx.fillStyle = det.color;
      ctx.fillRect(pillX, pillY, pillW, pillH);

      ctx.fillStyle = '#FFFFFF';
      ctx.textBaseline = 'top';
      ctx.fillText(labelText, pillX + padX, pillY + padY);

      ctx.restore();
    });
  }, [imageElement, detections, showOverlay, selectedDetectionId]);

  // Allow clicking directly on a bounding box inside the canvas to select it
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const clickX = (e.clientX - rect.left) * scaleX;
    const clickY = (e.clientY - rect.top) * scaleY;

    // Find topmost (smallest area) box containing the click point
    const hits = detections.filter((d) => {
      const [x1, y1, x2, y2] = d.bbox;
      return clickX >= x1 && clickX <= x2 && clickY >= y1 && clickY <= y2;
    });

    if (hits.length === 0) {
      onSelectDetection(null);
      return;
    }

    hits.sort((a, b) => {
      const areaA = (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1]);
      const areaB = (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]);
      return areaA - areaB;
    });

    const targetId = hits[0].id;
    onSelectDetection(selectedDetectionId === targetId ? null : targetId);
  };

  const handleDownloadAnnotatedImage = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dataUrl = canvas.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = dataUrl;
    a.download = `detected-${imageName.replace(/\.[^/.]+$/, '')}.png`;
    a.click();
  };

  return (
    <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl overflow-hidden shadow-sm">
      {/* Workspace Action Bar */}
      <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-800 flex flex-wrap items-center justify-between gap-2 bg-slate-50/70 dark:bg-slate-950/50">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-xs font-semibold text-slate-700 dark:text-slate-200 truncate max-w-[200px] sm:max-w-xs">
            {imageName}
          </span>
          <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-200/70 dark:bg-slate-800 text-slate-600 dark:text-slate-400">
            {imageElement.naturalWidth}×{imageElement.naturalHeight}px
          </span>
          <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
            {inferenceTimeMs} ms
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          <button
            type="button"
            onClick={() => setShowOverlay((prev) => !prev)}
            className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
              showOverlay
                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-600 dark:text-emerald-400'
                : 'bg-slate-100 dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300'
            }`}
          >
            {showOverlay ? (
              <>
                <Eye className="w-3.5 h-3.5" />
                <span>Boxes On</span>
              </>
            ) : (
              <>
                <EyeOff className="w-3.5 h-3.5" />
                <span>Boxes Off</span>
              </>
            )}
          </button>

          <button
            type="button"
            onClick={() => setSideBySideView((prev) => !prev)}
            className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
              sideBySideView
                ? 'bg-sky-500/10 border-sky-500/30 text-sky-600 dark:text-sky-400'
                : 'bg-slate-100 dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300'
            }`}
          >
            {sideBySideView ? (
              <>
                <Maximize2 className="w-3.5 h-3.5" />
                <span>Single View</span>
              </>
            ) : (
              <>
                <Columns className="w-3.5 h-3.5" />
                <span>Compare Original</span>
              </>
            )}
          </button>

          <button
            type="button"
            onClick={handleDownloadAnnotatedImage}
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200 border border-slate-200 dark:border-slate-700"
          >
            <Download className="w-3.5 h-3.5 text-emerald-500" />
            <span>Save PNG</span>
          </button>

          <input
            ref={replaceInputRef}
            type="file"
            accept="image/jpeg,image/jpg,image/png,image/webp"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) {
                onReplaceImage(file);
                e.target.value = '';
              }
            }}
            className="hidden"
          />
          <button
            type="button"
            onClick={() => replaceInputRef.current?.click()}
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200 border border-slate-200 dark:border-slate-700"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Replace</span>
          </button>

          <button
            type="button"
            onClick={onClearImage}
            className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-medium bg-rose-500/10 hover:bg-rose-500/20 text-rose-600 dark:text-rose-400 border border-rose-500/20"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear</span>
          </button>
        </div>
      </div>

      {/* Main Viewport */}
      <div
        className={`p-4 bg-slate-950/40 grid gap-4 items-center justify-center ${
          sideBySideView ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1'
        }`}
      >
        {sideBySideView && (
          <div className="space-y-1.5 text-center">
            <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400">
              Original Input Image
            </span>
            <div className="rounded-xl overflow-hidden border border-slate-800 bg-slate-950 flex items-center justify-center max-h-[540px]">
              <img
                src={imageElement.src}
                alt="Original input"
                className="max-h-[540px] w-auto object-contain"
              />
            </div>
          </div>
        )}

        <div className="space-y-1.5 text-center">
          {sideBySideView && (
            <span className="text-[11px] font-mono uppercase tracking-wider text-emerald-400">
              Annotated Computer Vision Output ({detections.length} objects)
            </span>
          )}
          <div className="rounded-xl overflow-hidden border border-slate-800 bg-slate-950 flex items-center justify-center max-h-[540px]">
            <canvas
              ref={canvasRef}
              onClick={handleCanvasClick}
              title="Click any bounding box to highlight its inspection card"
              className="max-h-[540px] w-auto max-w-full object-contain cursor-crosshair"
            />
          </div>
        </div>
      </div>
    </div>
  );
};
