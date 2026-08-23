"use client";

import React, { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { SyncingStandby } from "./SyncingStandby";

interface SyncGuardProps {
  children: React.ReactNode;
}

const STORAGE_KEY = "sarmayasaaz_is_syncing";
const STARTED_KEY = "sarmayasaaz_sync_started_at";

export function SyncGuard({ children }: SyncGuardProps) {
  // Synchronously initialize syncing state from localStorage to prevent 1-3s flash on page reload
  const [cachedSyncing, setCachedSyncing] = useState<boolean>(() => {
    if (typeof window === "undefined") return false;
    return localStorage.getItem(STORAGE_KEY) === "true";
  });
  const [cachedStartedAt, setCachedStartedAt] = useState<string>(() => {
    if (typeof window === "undefined") return "";
    return localStorage.getItem(STARTED_KEY) || "";
  });

  const { data: statusData, isFetched } = useQuery({
    queryKey: ["system", "sync-status"],
    queryFn: async () => {
      try {
        const res = await fetch("/api/data/freshness");
        if (!res.ok) return { is_syncing: false };
        return await res.json();
      } catch {
        return { is_syncing: false };
      }
    },
    refetchInterval: 3000,
    staleTime: 1000,
  });

  // Keep localStorage & cached state updated when fresh data arrives
  useEffect(() => {
    if (statusData && typeof window !== "undefined") {
      const syncing = Boolean(statusData.is_syncing);
      const started = statusData.started_at || "";
      setCachedSyncing(syncing);
      setCachedStartedAt(started);
      localStorage.setItem(STORAGE_KEY, syncing ? "true" : "false");
      if (started) localStorage.setItem(STARTED_KEY, started);
      else localStorage.removeItem(STARTED_KEY);
    }
  }, [statusData]);

  // Use fresh backend query status once fetched, fallback to cachedSyncing during initial query load
  const isSyncing = isFetched ? Boolean(statusData?.is_syncing) : cachedSyncing;
  const startedAt = isFetched ? statusData?.started_at : cachedStartedAt;

  if (isSyncing) {
    return (
      <SyncingStandby
        isDemoMode={false}
        progress={statusData?.progress ?? 0}
        currentStep={statusData?.step ?? statusData?.current_step ?? "Updating data..."}
        startedAt={startedAt}
      />
    );
  }

  return <>{children}</>;
}

