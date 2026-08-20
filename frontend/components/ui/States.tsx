"use client";

import { AlertTriangle, Inbox, ServerCrash } from "lucide-react";
import { cn } from "@/lib/format";

export function Skeleton({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-md bg-surface-high",
        "after:absolute after:inset-0 after:-translate-x-full after:animate-shimmer",
        "after:bg-gradient-to-r after:from-transparent after:via-black/5 after:to-transparent",
        "dark:after:via-white/5",
        className,
      )}
    />
  );
}

export function ErrorState({
  error,
  hint,
}: {
  error: unknown;
  hint?: string;
}) {
  const message = error instanceof Error ? error.message : "Something went wrong";
  const status = (error as { status?: number })?.status;
  const unreachable = status === 0;

  return (
    <div className="card-pad flex flex-col items-center gap-3 py-10 text-center">
      {unreachable ? (
        <ServerCrash size={28} className="text-neg" />
      ) : (
        <AlertTriangle size={28} className="text-warn" />
      )}
      <p className="max-w-md text-sm font-medium">{message}</p>
      {unreachable && (
        <pre className="inset px-3 py-2 text-left text-xs text-dim">
          uv run uvicorn backend.main:app --port 8000
        </pre>
      )}
      {hint && !unreachable && <p className="max-w-md text-xs text-dim">{hint}</p>}
    </div>
  );
}

export function EmptyState({ title, detail }: { title: string; detail?: string }) {
  return (
    <div className="card-pad flex flex-col items-center gap-2 py-10 text-center">
      <Inbox size={26} className="text-dim" />
      <p className="text-sm font-medium">{title}</p>
      {detail && <p className="max-w-md text-xs text-dim">{detail}</p>}
    </div>
  );
}

/**
 * Renders a value that the system genuinely does not have, rather than
 * substituting a plausible-looking number.
 */
export function NotMeasured({ label = "not measured" }: { label?: string }) {
  return (
    <span className="text-xs italic text-dim" title="This metric was never recorded during training">
      {label}
    </span>
  );
}
