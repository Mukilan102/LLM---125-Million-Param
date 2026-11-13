"use client";

import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  Bot,
  Cpu,
  Menu,
  MessageSquarePlus,
  RefreshCw,
  Send,
  Settings2,
  Trash2,
  X,
} from "lucide-react";
import {
  fetchHealth,
  HealthResponsePayload,
  sendChatMessage,
} from "@/lib/api";
import MessageBubble, { UIMessage } from "./MessageBubble";
import SettingsPanel, { GenerationSettings } from "./SettingsPanel";
import TypingIndicator from "./TypingIndicator";

const DEFAULT_SETTINGS: GenerationSettings = {
  temperature: 0.7,
  topK: 40,
  maxTokens: 256,
  autoLength: true,
};

const EXAMPLE_PROMPTS = [
  {
    label: "Short Factual",
    prompt: "What is the capital of Japan?",
  },
  {
    label: "Explicit Brief",
    prompt: "Explain briefly what an API is.",
  },
  {
    label: "Moderate General",
    prompt: "What is overfitting in machine learning?",
  },
  {
    label: "Detailed Step-by-Step",
    prompt:
      "Explain step by step how a transformer language model generates text.",
  },
];

function formatTime(): string {
  return new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [settings, setSettings] =
    useState<GenerationSettings>(DEFAULT_SETTINGS);
  const [health, setHealth] = useState<HealthResponsePayload | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const refreshHealth = async () => {
    try {
      const data = await fetchHealth();
      setHealth(data);
      setHealthError(null);
    } catch (err) {
      setHealth(null);
      setHealthError(
        err instanceof Error ? err.message : "Backend unreachable"
      );
    }
  };

  useEffect(() => {
    refreshHealth();
    const interval = setInterval(refreshHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const submitPrompt = async (promptText: string) => {
    const trimmed = promptText.trim();
    if (!trimmed || isLoading) return;

    setError(null);

    const userMsg: UIMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: trimmed,
      timestamp: formatTime(),
    };

    const historyPayload = messages.slice(-4).map((m) => ({
      role: m.role,
      content: m.content,
    }));

    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);

    try {
      const res = await sendChatMessage({
        message: trimmed,
        history: historyPayload.length > 0 ? historyPayload : undefined,
        temperature: settings.temperature,
        top_k: settings.topK,
        max_tokens: settings.maxTokens,
        auto_length: settings.autoLength,
      });

      const assistantMsg: UIMessage = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: res.response,
        timestamp: formatTime(),
        meta: {
          tokensGenerated: res.tokens_generated,
          tokenBudget: res.token_budget,
          lengthCategory: res.length_category,
          stoppedReason: res.stopped_reason,
          generationTimeMs: res.generation_time_ms,
          modelVariant: res.model_variant,
        },
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      const msg =
        err instanceof Error
          ? err.message
          : "Failed to generate a response from GPT-2 backend.";
      setError(msg);
    } finally {
      setIsLoading(false);
      textareaRef.current?.focus();
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    await submitPrompt(input);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void submitPrompt(input);
    }
  };

  const handleClearConversation = () => {
    setMessages([]);
    setError(null);
    textareaRef.current?.focus();
  };

  const isHealthy = health?.status === "ok" && health?.model_loaded;

  return (
    <div className="flex h-screen w-full overflow-hidden bg-surface-950 text-slate-100">
      {/* Mobile sidebar backdrop */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/60 backdrop-blur-sm lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Left Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-80 flex-col border-r border-slate-800/80 bg-surface-950 p-4 transition-transform duration-200 lg:static lg:translate-x-0 ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-emerald-500/15 border border-emerald-500/30 text-emerald-400">
              <Bot className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-sm font-semibold text-white">GPT-2 124M</h1>
              <p className="text-[11px] text-slate-400">
                Custom PyTorch Chat
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={() => setSidebarOpen(false)}
            className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-800 hover:text-white lg:hidden"
            aria-label="Close sidebar"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <button
          type="button"
          onClick={handleClearConversation}
          disabled={messages.length === 0 && !error}
          className="mb-4 flex w-full items-center justify-center gap-2 rounded-xl border border-slate-700/80 bg-surface-900 px-3 py-2.5 text-sm font-medium text-slate-200 transition hover:border-emerald-500/50 hover:bg-surface-800 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <MessageSquarePlus className="h-4 w-4 text-emerald-400" />
          <span>New Conversation</span>
        </button>

        <div className="flex-1 space-y-4 overflow-y-auto pr-1">
          <SettingsPanel
            settings={settings}
            onChange={setSettings}
            onReset={() => setSettings(DEFAULT_SETTINGS)}
          />

          {/* Backend / Model Status Card */}
          <div className="space-y-2.5 rounded-xl border border-slate-800 bg-surface-900/70 p-4 text-xs">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 font-semibold uppercase tracking-wider text-slate-300">
                <Cpu className="h-3.5 w-3.5 text-emerald-400" />
                <span>Model Runtime</span>
              </div>
              <button
                type="button"
                onClick={refreshHealth}
                title="Refresh backend health"
                className="rounded p-1 text-slate-400 hover:bg-slate-800 hover:text-emerald-400"
              >
                <RefreshCw className="h-3.5 w-3.5" />
              </button>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-slate-400">Status</span>
              <span
                className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] font-medium ${
                  isHealthy
                    ? "bg-emerald-500/15 text-emerald-300"
                    : "bg-rose-500/15 text-rose-300"
                }`}
              >
                <span
                  className={`h-1.5 w-1.5 rounded-full ${
                    isHealthy ? "bg-emerald-400" : "bg-rose-400"
                  }`}
                />
                {isHealthy ? "Ready" : healthError ? "Offline" : "Degraded"}
              </span>
            </div>

            {health && (
              <>
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Variant</span>
                  <span className="font-mono text-slate-200">
                    {health.model_variant}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Device</span>
                  <span className="font-mono uppercase text-slate-200">
                    {health.device}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Context</span>
                  <span className="font-mono text-slate-200">
                    {health.context_length} tokens
                  </span>
                </div>
              </>
            )}

            {healthError && (
              <p className="text-[11px] text-rose-300">{healthError}</p>
            )}
          </div>
        </div>
      </aside>

      {/* Main Chat Column */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Top Bar */}
        <header className="flex h-14 shrink-0 items-center justify-between border-b border-slate-800/80 bg-surface-950/90 px-4 backdrop-blur">
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => setSidebarOpen(true)}
              className="rounded-lg border border-slate-800 p-2 text-slate-300 hover:bg-surface-800 lg:hidden"
              aria-label="Open settings"
            >
              <Menu className="h-4 w-4" />
            </button>
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold text-white">
                Custom GPT-2 Chat
              </span>
              {health?.model_variant && (
                <span className="hidden rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-0.5 text-[11px] font-medium text-emerald-300 sm:inline-block">
                  {health.model_variant} ({health.device})
                </span>
              )}
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setSidebarOpen(true)}
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-800 bg-surface-900 px-2.5 py-1.5 text-xs text-slate-300 hover:bg-surface-800 lg:hidden"
            >
              <Settings2 className="h-3.5 w-3.5" />
              <span>Settings</span>
            </button>

            <button
              type="button"
              onClick={handleClearConversation}
              disabled={messages.length === 0 && !error}
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-800 bg-surface-900 px-3 py-1.5 text-xs font-medium text-slate-300 transition hover:border-rose-500/40 hover:text-rose-300 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <Trash2 className="h-3.5 w-3.5" />
              <span>Clear</span>
            </button>
          </div>
        </header>

        {/* Messages Scroll Area */}
        <main className="flex-1 overflow-y-auto px-4 py-6 sm:px-6">
          <div className="mx-auto flex max-w-3xl flex-col gap-5">
            {messages.length === 0 ? (
              <div className="my-auto flex flex-col items-center justify-center py-10 text-center">
                <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl border border-emerald-500/30 bg-emerald-500/10 text-emerald-400">
                  <Bot className="h-7 w-7" />
                </div>
                <h2 className="text-xl font-semibold text-white">
                  Chat with your custom GPT-2 124M
                </h2>
                <p className="mt-2 max-w-lg text-sm text-slate-400">
                  Powered directly by your PyTorch{" "}
                  <code className="rounded bg-surface-800 px-1.5 py-0.5 text-xs text-emerald-300">
                    GPTModel
                  </code>{" "}
                  with instruction fine-tuning, Top-K sampling, temperature
                  scaling, EOS stopping, and dynamic token budgets.
                </p>

                <div className="mt-8 grid w-full gap-3 sm:grid-cols-2">
                  {EXAMPLE_PROMPTS.map((item) => (
                    <button
                      key={item.label}
                      type="button"
                      onClick={() => void submitPrompt(item.prompt)}
                      className="flex flex-col items-start gap-1 rounded-xl border border-slate-800 bg-surface-900/80 p-3.5 text-left transition hover:border-emerald-500/50 hover:bg-surface-800"
                    >
                      <span className="text-[11px] font-semibold uppercase tracking-wider text-emerald-400">
                        {item.label}
                      </span>
                      <span className="text-sm text-slate-200">
                        {item.prompt}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((msg) => (
                <MessageBubble key={msg.id} message={msg} />
              ))
            )}

            {isLoading && <TypingIndicator />}

            {error && (
              <div className="flex items-start gap-3 rounded-xl border border-rose-500/40 bg-rose-950/40 p-3.5 text-sm text-rose-200">
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-rose-400" />
                <div className="flex-1">
                  <p className="font-medium text-rose-300">Generation Error</p>
                  <p className="mt-0.5 text-xs text-rose-200/90">{error}</p>
                </div>
                <button
                  type="button"
                  onClick={() => setError(null)}
                  className="rounded p-1 text-rose-300 hover:bg-rose-900/50"
                  aria-label="Dismiss error"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </main>

        {/* Message Input Bar */}
        <footer className="shrink-0 border-t border-slate-800/80 bg-surface-950 px-4 py-3 sm:px-6">
          <form
            onSubmit={handleSubmit}
            className="mx-auto flex max-w-3xl items-end gap-2.5 rounded-2xl border border-slate-800 bg-surface-900 p-2 focus-within:border-emerald-500/60"
          >
            <textarea
              ref={textareaRef}
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask your GPT-2 model anything... (Enter to send, Shift+Enter for newline)"
              disabled={isLoading}
              className="max-h-40 min-h-[44px] flex-1 resize-none bg-transparent px-2.5 py-2.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none disabled:opacity-60"
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              aria-label="Send message"
              className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-emerald-600 text-white transition hover:bg-emerald-500 disabled:cursor-not-allowed disabled:bg-slate-800 disabled:text-slate-500"
            >
              <Send className="h-4 w-4" />
            </button>
          </form>
          <p className="mx-auto mt-1.5 max-w-3xl text-center text-[11px] text-slate-500">
            Local inference only • Temp {settings.temperature.toFixed(2)} • Top-K{" "}
            {settings.topK} •{" "}
            {settings.autoLength
              ? `Auto budget (cap ${settings.maxTokens})`
              : `Fixed cap ${settings.maxTokens}`}
          </p>
        </footer>
      </div>
    </div>
  );
}

