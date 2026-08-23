"use client";

import React, { useEffect, useState } from "react";
import Image from "next/image";
import { RefreshCw, CheckCircle2, Cpu, Database, Sparkles, Activity } from "lucide-react";
import logoDark from "@/public/logo-dark.png";
import logoLight from "@/public/logo-light.png";

interface SyncingStandbyProps {
  progress?: number;
  currentStep?: string;
  onRefreshFinished?: () => void;
  isDemoMode?: boolean;
}

const SYNC_STEPS = [
  { id: "init", text: "Initializing data procurement pipeline...", icon: Cpu, type: "OK" },
  { id: "collect", text: "Fetching daily bars (PSX, MUFAP, Crypto, Commodities)...", icon: Database, type: "OK" },
  { id: "features", text: "Recalculating technical indicators & macro proxies...", icon: Activity, type: "SYNC" },
  { id: "models", text: "Running multi-horizon AI ensemble predictions...", icon: Sparkles, type: "MODEL" },
  { id: "snapshot", text: "Rebuilding universe snapshot & refreshing cache...", icon: RefreshCw, type: "DONE" },
];

export function SyncingStandby({
  progress: externalProgress,
  currentStep: externalStep,
  onRefreshFinished,
  isDemoMode = false,
}: SyncingStandbyProps) {
  const [internalProgress, setInternalProgress] = useState(18);
  const [activeStepIndex, setActiveStepIndex] = useState(1);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  // Demo loop if running in standalone preview or demo mode
  useEffect(() => {
    if (!isDemoMode && externalProgress !== undefined) return;

    const timer = setInterval(() => {
      setElapsedSeconds((prev) => prev + 1);
    }, 1000);

    const progressTimer = setInterval(() => {
      setInternalProgress((prev) => {
        if (prev >= 95) {
          if (onRefreshFinished) onRefreshFinished();
          return 95;
        }
        const next = prev + Math.floor(Math.random() * 8) + 4;
        if (next > 25 && next < 50) setActiveStepIndex(1);
        else if (next >= 50 && next < 75) setActiveStepIndex(2);
        else if (next >= 75 && next < 90) setActiveStepIndex(3);
        else if (next >= 90) setActiveStepIndex(4);
        return Math.min(next, 95);
      });
    }, 1200);

    return () => {
      clearInterval(timer);
      clearInterval(progressTimer);
    };
  }, [isDemoMode, externalProgress, onRefreshFinished]);

  const displayProgress = externalProgress !== undefined ? externalProgress : internalProgress;
  const activeIndex = externalProgress !== undefined
    ? (displayProgress >= 90 ? 4 : displayProgress >= 75 ? 3 : displayProgress >= 50 ? 2 : displayProgress >= 25 ? 1 : 0)
    : activeStepIndex;
  const currentStatusStep = SYNC_STEPS[Math.min(activeIndex, SYNC_STEPS.length - 1)];
  const currentStatusText = externalStep || (currentStatusStep ? currentStatusStep.text : "Updating market data...");


  return (
    <div className="fixed inset-0 z-[99999] h-screen w-screen flex flex-col items-center justify-center overflow-hidden bg-[#0b1326] px-4 py-8 text-slate-100 antialiased selection:bg-indigo-500/30 selection:text-indigo-200">
      {/* Grid Pattern & Radiant Ambient Glow */}
      <div
        className="absolute inset-0 pointer-events-none opacity-20"
        style={{
          backgroundImage: `
            linear-gradient(to right, rgba(129, 140, 248, 0.08) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(129, 140, 248, 0.08) 1px, transparent 1px)
          `,
          backgroundSize: "36px 36px",
        }}
      />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[550px] h-[550px] bg-indigo-600/10 rounded-full blur-[130px] pointer-events-none" />
      <div className="absolute top-1/3 left-1/2 -translate-x-1/2 w-[350px] h-[350px] bg-emerald-500/5 rounded-full blur-[100px] pointer-events-none" />

      {/* Main Lockout Container */}
      <div className="relative z-10 w-full max-w-xl mx-auto flex flex-col items-center text-center space-y-8">
        
        {/* SarmayaSaaz Official Brand Logo Container */}
        <div className="flex flex-col items-center space-y-3">
          <div className="relative group">
            {/* Outer radar pulse rings */}
            <div className="absolute -inset-3 rounded-2xl bg-gradient-to-r from-indigo-500 via-purple-500 to-emerald-400 opacity-30 blur-xl group-hover:opacity-50 transition duration-500 animate-pulse" />
            <div className="relative px-6 py-4 rounded-2xl bg-[#131b2e] border border-indigo-500/30 shadow-2xl flex items-center justify-center overflow-hidden backdrop-blur-md">
              <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/10 via-transparent to-emerald-500/10" />
              
              {/* SarmayaSaaz Official Logo Images */}
              <Image
                src={logoLight}
                alt="SarmayaSaaz"
                priority
                className="h-9 w-auto dark:hidden"
              />
              <Image
                src={logoDark}
                alt="SarmayaSaaz"
                priority
                className="hidden h-9 w-auto dark:block"
              />

              {/* Corner accent LED */}
              <span className="absolute top-2 right-2 flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
              </span>
            </div>
          </div>

          <p className="text-xs font-medium uppercase tracking-widest text-indigo-300/70">
            AI Financial Forecasting System
          </p>
        </div>

        {/* Sync Status Card */}
        <div className="w-full bg-[#131b2e]/90 backdrop-blur-xl rounded-2xl border border-indigo-500/20 p-6 sm:p-8 shadow-2xl shadow-black/50 space-y-6 text-left">
          
          {/* Status Header */}
          <div className="flex items-center justify-between border-b border-indigo-500/10 pb-4">
            <div>
              <h2 className="text-lg font-semibold text-slate-100 flex items-center gap-2">
                <RefreshCw className="w-4 h-4 text-indigo-400 animate-spin" />
                Syncing Financial Intelligence
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Market session update & model tuning in progress
              </p>
            </div>
            <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-xs font-mono text-indigo-300">
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />
              {displayProgress}%
            </div>
          </div>

          {/* Progress Bar with Light Shimmer */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs font-mono text-slate-400">
              <span>Sync Completion</span>
              <span>{displayProgress}%</span>
            </div>
            <div className="h-2.5 w-full bg-[#060e20] rounded-full overflow-hidden p-0.5 border border-indigo-500/20 relative">
              <div
                className="h-full bg-gradient-to-r from-indigo-500 via-purple-500 to-emerald-400 rounded-full transition-all duration-500 shadow-[0_0_12px_rgba(129,140,248,0.6)] relative overflow-hidden"
                style={{ width: `${displayProgress}%` }}
              >
                {/* Continuous Shimmer effect */}
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent -translate-x-full animate-[shimmer_1.5s_infinite]" />
              </div>
            </div>
          </div>

          {/* Active Status Banner */}
          <div className="flex items-center gap-3 p-3 rounded-xl bg-[#060e20]/80 border border-indigo-500/15 text-xs">
            <div className="w-2 h-2 rounded-full bg-indigo-400 animate-pulse flex-shrink-0" />
            <span className="font-mono text-indigo-200 truncate">{currentStatusText}</span>
          </div>

          {/* Terminal Execution Log Window */}
          <div className="bg-[#060e20] rounded-xl border border-slate-800/80 p-4 font-mono text-[11px] leading-relaxed text-slate-300 shadow-inner space-y-2">
            {/* Terminal Window Control Bar */}
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-2 mb-2">
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-rose-500/80 inline-block" />
                <span className="w-2.5 h-2.5 rounded-full bg-amber-500/80 inline-block" />
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-500/80 inline-block" />
                <span className="text-[10px] text-slate-500 ml-2">SYSTEM EXECUTION STREAM</span>
              </div>
              <span className="text-[10px] text-indigo-400/80">{elapsedSeconds}s elapsed</span>
            </div>

            {/* Log Stream List */}
            <ul className="space-y-1.5 overflow-hidden max-h-36">
              {SYNC_STEPS.map((step, idx) => {
                const isCompleted = idx < activeIndex;
                const isCurrent = idx === activeIndex;
                const isPending = idx > activeIndex;

                return (
                  <li
                    key={step.id}
                    className={`flex items-center gap-2 transition-opacity duration-300 ${
                      isPending ? "opacity-35" : "opacity-100"
                    }`}
                  >
                    {isCompleted && (
                      <span className="text-emerald-400 font-bold flex items-center gap-1">
                        <CheckCircle2 className="w-3 h-3 inline" /> [OK]
                      </span>
                    )}
                    {isCurrent && (
                      <span className="text-indigo-400 font-bold flex items-center gap-1 animate-pulse">
                        <RefreshCw className="w-3 h-3 inline animate-spin" /> [RUNNING]
                      </span>
                    )}
                    {isPending && (
                      <span className="text-slate-600 font-bold">[QUEUED]</span>
                    )}

                    <span className={isCurrent ? "text-slate-100 font-medium" : "text-slate-400"}>
                      {step.text}
                    </span>
                  </li>
                );
              })}
            </ul>
          </div>

          {/* Guidance Notice */}
          <div className="text-center pt-1 space-y-1">
            <p className="text-xs text-slate-400">
              The platform is temporarily locked while fresh market data & predictions synchronize.
              This usually takes under 30 seconds.
            </p>
          </div>

        </div>

        {/* System Footnote */}
        <div className="flex items-center justify-center gap-4 text-[11px] text-slate-500 font-mono">
          <span className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-ping inline-block" />
            Maintenance Mode Active
          </span>
          <span>•</span>
          <span>Automatic Unlock on Completion</span>
        </div>

      </div>
    </div>
  );
}
