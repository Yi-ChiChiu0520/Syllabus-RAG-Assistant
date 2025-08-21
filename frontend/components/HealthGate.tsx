"use client";
import { useEffect, useState } from "react";

export default function HealthGate({ children }: { children: React.ReactNode }) {
  const [ready, setReady] = useState(false);
  const [log, setLog] = useState<string[]>([]);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      let delay = 500;
      for (let i = 1; i <= 10; i++) {
        try {
          const r = await fetch("/api/health", { cache: "no-store" });
          if (r.ok) {
            setLog((l) => [...l, "✅ Backend is ready."]);
            if (!cancelled) setReady(true);
            return;
          }
          setLog((l) => [...l, `Attempt ${i}: ${r.status} ${r.statusText}`]);
        } catch (e: any) {
          setLog((l) => [...l, `Attempt ${i}: ${e?.message ?? e}`]);
        }
        await new Promise((r) => setTimeout(r, delay));
        delay = Math.min(delay * 2, 4000);
      }
      setLog((l) => [...l, "❌ Backend not available within the wait window."]);
    };
    poll();
    return () => { cancelled = true; };
  }, []);

  if (!ready) {
    return (
      <main className="min-h-screen max-w-3xl mx-auto px-4 py-10">
        <h1 className="text-3xl font-semibold mb-3">📚 Syllabus RAG Assistant</h1>
        <p className="text-sm text-muted">🚀 Initializing backend (loading models)…</p>
        <pre className="mt-4 text-xs bg-zinc-950 border border-zinc-800/60 rounded-xl p-3">{log.join("\n")}</pre>
      </main>
    );
  }
  return <>{children}</>;
}
