"use client";
import { useEffect, useState } from "react";

export type ModelCfg = { modelTypeKey: "openai" | "ollama"; modelName: string };

export default function ModelSelector({
  cfg, onChange,
}: { cfg: ModelCfg; onChange: (v: ModelCfg) => void }) {
  const [models, setModels] = useState<{ openai_available: boolean; ollama_models: string[] }>({
    openai_available: false, ollama_models: ["llama3.1:8b"],
  });

  useEffect(() => {
    fetch("/api/models").then(r => r.json()).then(setModels).catch(() => {});
  }, []);

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4 text-sm">
        {models.openai_available && (
          <label className="flex items-center gap-2">
            <input
              type="radio"
              name="modelType"
              checked={cfg.modelTypeKey === "openai"}
              onChange={() => onChange({ ...cfg, modelTypeKey: "openai", modelName: "gpt-4o" })}
            />
            OpenAI GPT-4o
          </label>
        )}
        {models.ollama_models?.length > 0 && (
          <label className="flex items-center gap-2">
            <input
              type="radio"
              name="modelType"
              checked={cfg.modelTypeKey === "ollama"}
              onChange={() => onChange({ ...cfg, modelTypeKey: "ollama" })}
            />
            Local Ollama Model
          </label>
        )}
      </div>

      {cfg.modelTypeKey === "ollama" && (
        <select
          className="select"
          value={cfg.modelName}
          onChange={(e) => onChange({ ...cfg, modelName: e.target.value })}
        >
          {models.ollama_models.map((m) => <option key={m}>{m}</option>)}
        </select>
      )}

      <div className="badge">
        <span className="w-2 h-2 rounded-full bg-emerald-500 mr-1"></span>
        Active Model: {cfg.modelName}
      </div>
    </div>
  );
}
