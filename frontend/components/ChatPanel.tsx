"use client";
import { useState } from "react";
import type { ModelCfg } from "./ModelSelector";
import { Send } from "lucide-react";

export default function ChatPanel({
  cfg, activeCollection,
}: { cfg: ModelCfg; activeCollection: string | null }) {
  const [messages, setMessages] = useState<{ role: "user" | "assistant"; content: string; details?: any }[]>([]);
  const [input, setInput] = useState("");

  const ask = async () => {
    if (!input.trim() || !activeCollection) return;
    const q = input.trim();
    setMessages((m) => [...m, { role: "user", content: q }]);
    setInput("");

    const payload = {
      query: q,
      collection_name: activeCollection,
      model_type: cfg.modelTypeKey,
      model_name: cfg.modelName,
    };

    const r = await fetch("/api/query", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!r.ok) {
      setMessages((m) => [...m, { role: "assistant", content: `❌ Error ${r.status}` }]);
      return;
    }
    const data = await r.json();
    setMessages((m) => [...m, { role: "assistant", content: data.response, details: data }]);
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <input
          className="input"
          placeholder="Ask me anything about your course syllabi…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e)=> e.key==="Enter" && ask()}
        />
        <button className="btn btn-primary" onClick={ask}>
          <Send size={16} className="mr-2"/> Ask
        </button>
      </div>

      <div className="space-y-4">
        {messages.map((m, i) => (
          <div key={i} className="card">
            <div className="card-header">{m.role === "user" ? "You" : "Assistant"}</div>
            <div className="card-body">
              <p className="whitespace-pre-wrap text-sm">{m.content}</p>
              {m.details?.best_chunk && (
                <details className="mt-3">
                  <summary className="cursor-pointer text-sm text-emerald-400">View Processing Details</summary>
                  <div className="mt-2 text-xs space-y-1 text-zinc-300">
                    <div>Optimized Query: <code>{m.details.cleaned_query}</code></div>
                    <div>Best ID: {m.details.best_id} | Score: {m.details.best_score?.toFixed?.(4)}</div>
                    <div className="mt-2">
                      <div className="mb-1 text-zinc-400">Source Content Used:</div>
                      <textarea className="w-full h-40 p-2 rounded-xl bg-zinc-950 border border-zinc-800/60" readOnly
                                value={m.details.best_chunk}/>
                    </div>
                  </div>
                </details>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
