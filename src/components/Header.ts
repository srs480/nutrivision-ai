import React from 'react';
import {
  Cpu,
  ShieldCheck,
  Sun,
  Moon,
  Sparkles,
  CheckCircle2,
  AlertTriangle,
  Loader2,
} from 'lucide-react';
import { ModelStatus } from '../types/detection';

interface HeaderProps {
  darkMode: boolean;
  onToggleDarkMode: () => void;
  modelStatus: ModelStatus;
  backend: 'webgpu' | 'wasm';
  modelSource: string;
}

export const Header: React.FC<HeaderProps> = ({
  darkMode,
  onToggleDarkMode,
  modelStatus,
  backend,
  modelSource,
}) => {
  return (
    <header className="sticky top-0 z-30 border-b border-slate-200 dark:border-slate-800/80 bg-white/85 dark:bg-slate-950/85 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center shadow-lg shadow-emerald-500/20">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-bold tracking-tight text-slate-900 dark:text-white">
                NutriVision <span className="text-emerald-500">CV</span>
              </h1>
              <span className="hidden sm:inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-medium bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
                100% Local ONNX
              </span>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 hidden sm:block">
              Multi-Object Food Detection & Localization Engine (Zero External APIs)
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2.5 sm:gap-3">
          {/* Privacy / Offline Guarantee Badge */}
          <div
            className="hidden md:flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-xs text-slate-600 dark:text-slate-300"
            title={`Source: ${modelSource || 'Initializing'}`}
          >
            <ShieldCheck className="w-4 h-4 text-emerald-500 shrink-0" />
            <span>Browser Edge Inference</span>
          </div>

          {/* Hardware Acceleration Badge */}
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-xs font-mono">
            <Cpu className="w-3.5 h-3.5 text-sky-500 shrink-0" />
            <span className="uppercase font-semibold text-slate-700 dark:text-slate-200">
              {backend}
            </span>
          </div>

          {/* Model Status Pill */}
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border bg-slate-50 dark:bg-slate-900 border-slate-200 dark:border-slate-800">
            {modelStatus === 'ready' && (
              <>
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
                <span className="text-emerald-600 dark:text-emerald-400">
                  Model Ready
                </span>
              </>
            )}
            {(modelStatus === 'downloading' ||
              modelStatus === 'checking_cache' ||
              modelStatus === 'initializing_session') && (
              <>
                <Loader2 className="w-3.5 h-3.5 text-amber-500 animate-spin" />
                <span className="text-amber-600 dark:text-amber-400">
                  Loading ONNX...
                </span>
              </>
            )}
            {modelStatus === 'error' && (
              <>
                <AlertTriangle className="w-3.5 h-3.5 text-rose-500" />
                <span className="text-rose-600 dark:text-rose-400">
                  Model Error
                </span>
              </>
            )}
          </div>

          {/* Theme Toggle */}
          <button
            onClick={onToggleDarkMode}
            className="p-2 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-100 dark:bg-slate-900 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-800 transition-colors"
            aria-label="Toggle color theme"
          >
            {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>
        </div>
      </div>
    </header>
  );
};