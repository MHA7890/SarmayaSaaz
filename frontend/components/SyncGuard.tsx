"use client";

import React from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api/client";
import { SyncingStandby } from "./SyncingStandby";

interface SyncGuardProps {
  children: React.ReactNode;
}

export function SyncGuard({ children }: SyncGuardProps) {
  // Poll backend sync & freshness status every 3 seconds
  const { data: statusData } = useQuery({
    queryKey: ["system", "sync-status"],
    queryFn: async () => {
      try {
        const res = await fetch("/api/data/freshness");
        if (!res.ok) return { is_syncing: false };
        const data = await res.json();
        return data;
      } catch {
        return { is_syncing: false };
      }
    },
    refetchInterval: 3000,
    staleTime: 2000,
  });

  const isSyncing = Boolean(statusData?.is_syncing);

  // If the backend is currently performing a data update:
  // Render ONLY the Standby Standalone Screen - lock out navigation, navbar, ticker and all pages
  if (isSyncing) {
    return <SyncingStandby isDemoMode={false} progress={statusData?.progress} currentStep={statusData?.step} />;
  }

  // Normal working hours / idle state: render normal website
  return <>{children}</>;
}
