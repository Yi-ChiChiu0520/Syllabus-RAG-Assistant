"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { ShieldCheck, Database, RefreshCw, Trash2, Upload, PlayCircle, Server, BookOpenText } from "lucide-react";
import HealthGate from "@/components/HealthGate";
import ModelSelector, { type ModelCfg } from "@/components/ModelSelector";
import CollectionsPanel from "@/components/CollectionsPanel";
import ChatPanel from "@/components/ChatPanel";

export default function Page() {
  const [cfg, setCfg] = useState<ModelCfg>({ modelTypeKey: "ollama", modelName: "llama3.1:8b" });
  const [activeCollection, setActiveCollection] = useState<string | null>(null);

  return (
    <HealthGate>
      <div className="min-h-screen">
        {/* Top banner */}
        <div className="sticky top-0 z-20 border-b border-zinc-800/60 bg-bg/70 backdrop-blur">
          <div className="mx-auto max-w-7xl px-4 py-4 flex items-center gap-3">
            <span className="text-2xl">📚</span>
            <div>
              <h1 className="text-2xl font-semibold">Syllabus RAG Assistant</h1>
              <p className="text-sm text-muted -mt-1">
                Ask questions about your course syllabi and get instant, accurate answers from your academic materials
              </p>
            </div>
            <span className="ml-auto badge"><Server size={14}/> Connected to backend API</span>
          </div>
        </div>

        {/* Content */}
        <div className="mx-auto max-w-7xl px-4 py-6 grid grid-cols-12 gap-6">
          {/* Sidebar */}
          <aside className="col-span-12 lg:col-span-4 space-y-6">
            <div className="card">
              <div className="card-header flex items-center gap-2">
                <ShieldCheck size={16}/> Model Configuration
              </div>
              <div className="card-body space-y-4">
                <ModelSelector cfg={cfg} onChange={setCfg}/>
              </div>
            </div>

            <CollectionsPanel
              onActiveChange={setActiveCollection}
              activeCollection={activeCollection}
            />
          </aside>

          {/* Main */}
          <main className="col-span-12 lg:col-span-8 space-y-6">
            <div className="card">
              <div className="card-header flex items-center gap-2">
                <BookOpenText size={16}/> Ask About Your Courses
              </div>
              <div className="card-body">
                <p className="text-sm text-muted mb-4">
                  Currently using: <span className="badge">{activeCollection ?? "—"}</span>
                </p>
                <ChatPanel cfg={cfg} activeCollection={activeCollection}/>
              </div>
            </div>

            {/* Tips */}
            <div className="card">
              <div className="card-header">Tips for Better Results</div>
              <div className="card-body">
                <ul className="text-sm text-zinc-300 list-disc pl-5 space-y-1">
                  <li>Be specific about what you’re looking for (dates, policies, requirements)</li>
                  <li>Ask about one topic at a time for clearer answers</li>
                  <li>Use academic terms when asking about course content</li>
                  <li>Check the source content in the details to verify information</li>
                </ul>
              </div>
            </div>
          </main>
        </div>
      </div>
    </HealthGate>
  );
}
