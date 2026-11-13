"use client";

import { useState } from "react";
import { Bot, Check, Copy, User } from "lucide-react";

export interface UIMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  meta?: {
    tokensGenerated: number;
    tokenBudget: number;
    lengthCategory: string;
    stoppedReason: string;
    generationTimeMs: number;
    modelVariant: string;
  };
}

interface MessageBubbleProps {
  message: UIMessage;
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === "user";

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      // Clipboard may be unavailable in non-HTTPS contexts
    }
  };

  return (
    <div
      className={`flex w-full gap-3 ${
        isUser ? "justify-end" : "justify-start"
      }`}
    >
      {!isUser && (
        <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-emerald-500/30 bg-emerald-500/10 text-emerald-400">
          <Bot className="h-4 w-4" />
        </div>
      )}

      <div
        className={`group relative max-w-[85%] sm:max-w-[78%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm ${
          isUser
            ? "bg-emerald-600 text-white rounded-br-sm"
            : "border border-slate-800 bg-surface-900 text-slate-100 rounded-bl-sm"
        }`}
      >
        <div className="whitespace-pre-wrap break-words">{message.content}</div>

        <div
          className={`mt-2 flex flex-wrap items-center gap-2 text-[11px] ${
            isUser ? "justify-end text-emerald-100/75" : "text-slate-400"
          }`}
        >
          <span>{message.timestamp}</span>

          {!isUser && message.meta && (
            <>
              <span className="text-slate-600">•</span>
              <span className="rounded bg-slate-800/90 px-1.5 py-0.5 font-mono text-[10px] text-slate-300">
                {message.meta.tokensGenerated}/{message.meta.tokenBudget} tok
              </span>
              <span className="rounded bg-emerald-950/60 border border-emerald-800/40 px-1.5 py-0.5 text-[10px] text-emerald-300">
                {message.meta.lengthCategory}
              </span>
              <span className="rounded bg-slate-800/90 px-1.5 py-0.5 text-[10px] text-slate-300">
                stop: {message.meta.stoppedReason}
              </span>
              <span>{Math.round(message.meta.generationTimeMs)} ms</span>
            </>
          )}

          {!isUser && (
            <button
              type="button"
              onClick={handleCopy}
              title="Copy response"
              className="ml-auto inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-slate-400 transition hover:bg-slate-800 hover:text-slate-200"
            >
              {copied ? (
                <>
                  <Check className="h-3 w-3 text-emerald-400" />
                  <span className="text-emerald-400">Copied</span>
                </>
              ) : (
                <>
                  <Copy className="h-3 w-3" />
                  <span>Copy</span>
                </>
              )}
            </button>
          )}
        </div>
      </div>

      {isUser && (
        <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-slate-700 bg-surface-800 text-slate-300">
          <User className="h-4 w-4" />
        </div>
      )}
    </div>
  );
}

