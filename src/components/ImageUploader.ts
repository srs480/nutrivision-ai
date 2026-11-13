import React, { useRef, useState } from 'react';
import {
  UploadCloud,
  Image as ImageIcon,
  AlertCircle,
  Sparkles,
} from 'lucide-react';
import { SAMPLE_TEST_IMAGES } from '../utils/imagePipeline';

interface ImageUploaderProps {
  onSelectFile: (file: File) => void;
  onSelectSampleUrl: (url: string, name: string) => void;
  error: string | null;
  isProcessing: boolean;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({
  onSelectFile,
  onSelectSampleUrl,
  error,
  isProcessing,
}) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const file = e.dataTransfer.files?.[0];
    if (file) {
      onSelectFile(file);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onSelectFile(file);
      e.target.value = '';
    }
  };

  return (
    <div className="space-y-4">
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isProcessing && fileInputRef.current?.click()}
        className={`relative group cursor-pointer rounded-2xl border-2 border-dashed transition-all p-8 sm:p-12 text-center ${
          isDragging
            ? 'border-emerald-500 bg-emerald-500/10'
            : 'border-slate-300 dark:border-slate-800 hover:border-emerald-500/60 bg-white dark:bg-slate-900/60 hover:bg-slate-50/80 dark:hover:bg-slate-900'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/jpg,image/png,image/webp"
          onChange={handleInputChange}
          className="hidden"
        />

        <div className="mx-auto w-14 h-14 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center mb-4 group-hover:scale-105 transition-transform">
          <UploadCloud className="w-7 h-7 text-emerald-500" />
        </div>

        <h3 className="text-base sm:text-lg font-semibold text-slate-900 dark:text-white mb-1">
          Drop a food image here, or <span className="text-emerald-500 underline">browse files</span>
        </h3>
        <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 max-w-md mx-auto">
          Supports JPG, JPEG, PNG, and WEBP up to 15 MB. Processed 100% locally inside your browser using ONNX Runtime Web.
        </p>

        <div className="mt-5 flex flex-wrap items-center justify-center gap-2 text-[11px] text-slate-500 dark:text-slate-400 font-mono">
          <span className="px-2.5 py-1 rounded-md bg-slate-100 dark:bg-slate-800">
            640×640 Letterbox Tensor
          </span>
          <span className="px-2.5 py-1 rounded-md bg-slate-100 dark:bg-slate-800">
            Class-Aware NMS
          </span>
          <span className="px-2.5 py-1 rounded-md bg-slate-100 dark:bg-slate-800">
            Multi-Instance Detection
          </span>
        </div>
      </div>

      {error && (
        <div className="flex items-start gap-2.5 p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-600 dark:text-rose-300 text-xs sm:text-sm">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-rose-500" />
          <span>{error}</span>
        </div>
      )}

      {/* Sample Test Images Bar */}
      <div className="bg-white dark:bg-slate-900/70 border border-slate-200 dark:border-slate-800 rounded-2xl p-4">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-3">
          <Sparkles className="w-3.5 h-3.5 text-emerald-500" />
          Quick Test Sample Food Images
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {SAMPLE_TEST_IMAGES.map((sample) => (
            <button
              key={sample.name}
              type="button"
              disabled={isProcessing}
              onClick={() => onSelectSampleUrl(sample.url, sample.name)}
              className="flex items-center gap-3 p-2 rounded-xl border border-slate-200 dark:border-slate-800 hover:border-emerald-500/50 bg-slate-50 dark:bg-slate-950/60 hover:bg-emerald-500/5 transition-all text-left group disabled:opacity-50"
            >
              <img
                src={sample.url}
                alt={sample.name}
                crossOrigin="anonymous"
                className="w-12 h-12 rounded-lg object-cover shrink-0 border border-slate-200 dark:border-slate-700"
              />
              <div className="min-w-0">
                <div className="text-xs font-semibold text-slate-800 dark:text-slate-200 truncate group-hover:text-emerald-500 flex items-center gap-1">
                  <ImageIcon className="w-3 h-3 shrink-0" />
                  <span className="truncate">{sample.name}</span>
                </div>
                <p className="text-[11px] text-slate-500 dark:text-slate-400 truncate">
                  {sample.description}
                </p>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};