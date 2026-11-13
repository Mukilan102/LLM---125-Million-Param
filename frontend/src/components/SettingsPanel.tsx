"use client";

import { Sliders, Sparkles } from "lucide-react";

export interface GenerationSettings {
  temperature: number;
  topK: number;
  maxTokens: number;
  autoLength: boolean;
}

interface SettingsPanelProps {
  settings: GenerationSettings;
  onChange: (next: GenerationSettings) => void;
  onReset: () => void;
}

export default function SettingsPanel({
  settings,
  onChange,
  onReset,
}: SettingsPanelProps) {
  return (
    <div className="space-y-4 rounded-xl border border-slate-800 bg-surface-900/90 p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-300">
          <Sliders className="h-3.5 w-3.5 text-emerald-400" />
          <span>Generation Settings</span>
        </div>
        <button
          type="button"
          onClick={onReset}
          className="text-xs text-slate-400 transition hover:text-emerald-400"
        >
          Reset
        </button>
      </div>

      {/* Temperature */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between text-xs">
          <label htmlFor="temp-slider" className="text-slate-300">
            Temperature
          </label>
          <span className="font-mono text-emerald-400">
            {settings.temperature.toFixed(2)}
          </span>
        </div>
        <input
          id="temp-slider"
          type="range"
          min={0}
          max={2}
          step={0.05}
          value={settings.temperature}
          onChange={(e) =>
            onChange({ ...settings, temperature: parseFloat(e.target.value) })
          }
          className="h-1.5 w-full cursor-pointer appearance-none rounded-lg bg-slate-700 accent-emerald-500"
        />
        <div className="flex justify-between text-[10px] text-slate-500">
          <span>0.0 (Greedy)</span>
          <span>0.7 (Balanced)</span>
          <span>2.0 (Creative)</span>
        </div>
      </div>

      {/* Top-K */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between text-xs">
          <label htmlFor="topk-slider" className="text-slate-300">
            Top-K Cutoff
          </label>
          <span className="font-mono text-emerald-400">{settings.topK}</span>
        </div>
        <input
          id="topk-slider"
          type="range"
          min={1}
          max={100}
          step={1}
          value={settings.topK}
          onChange={(e) =>
            onChange({ ...settings, topK: parseInt(e.target.value, 10) })
          }
          className="h-1.5 w-full cursor-pointer appearance-none rounded-lg bg-slate-700 accent-emerald-500"
        />
        <div className="flex justify-between text-[10px] text-slate-500">
          <span>1 (Strict)</span>
          <span>40 (Default)</span>
          <span>100 (Diverse)</span>
        </div>
      </div>

      {/* Max Tokens Ceiling */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between text-xs">
          <label htmlFor="maxtokens-slider" className="text-slate-300">
            Max Token Limit
          </label>
          <span className="font-mono text-emerald-400">
            {settings.maxTokens}
          </span>
        </div>
        <input
          id="maxtokens-slider"
          type="range"
          min={32}
          max={512}
          step={16}
          value={settings.maxTokens}
          onChange={(e) =>
            onChange({ ...settings, maxTokens: parseInt(e.target.value, 10) })
          }
          className="h-1.5 w-full cursor-pointer appearance-none rounded-lg bg-slate-700 accent-emerald-500"
        />
        <p className="text-[10px] leading-snug text-slate-500">
          Upper safety ceiling. The model stops naturally at{" "}
          <code className="text-slate-400">&lt;|endoftext|&gt;</code>.
        </p>
      </div>

      {/* Dynamic Token Budget Toggle */}
      <div className="flex items-start justify-between gap-3 rounded-lg border border-slate-800 bg-surface-950/60 p-3">
        <div className="space-y-0.5">
          <div className="flex items-center gap-1.5 text-xs font-medium text-slate-200">
            <Sparkles className="h-3.5 w-3.5 text-emerald-400" />
            <span>Dynamic Length Budget</span>
          </div>
          <p className="text-[11px] leading-snug text-slate-400">
            Adapts token limit automatically for brief, moderate, or detailed
            prompts.
          </p>
        </div>
        <button
          type="button"
          role="switch"
          aria-checked={settings.autoLength}
          onClick={() =>
            onChange({ ...settings, autoLength: !settings.autoLength })
          }
          className={`relative mt-0.5 inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors ${
            settings.autoLength ? "bg-emerald-600" : "bg-slate-700"
          }`}
        >
          <span
            className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow transition ${
              settings.autoLength ? "translate-x-4" : "translate-x-0"
            }`}
          />
        </button>
      </div>
    </div>
  );
}

