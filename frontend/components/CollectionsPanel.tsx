"use client";
import { useEffect, useMemo, useState } from "react";
import { Database, RefreshCw, Upload, Trash2 } from "lucide-react";

type Collection = { name: string; document_count: number; source_files?: Record<string, number> };

export default function CollectionsPanel({
  activeCollection, onActiveChange,
}: { activeCollection: string | null; onActiveChange: (v: string) => void; }) {
  const [collections, setCollections] = useState<Collection[]>([]);
  const [files, setFiles] = useState<FileList | null>(null);
  const [uploadMode, setUploadMode] = useState<"existing" | "new">("existing");
  const [newName, setNewName] = useState("my-syllabus-collection");
  const names = useMemo(() => collections.map((c) => c.name), [collections]);

  const refresh = () =>
    fetch("/api/collections").then(r => r.json()).then(setCollections).catch(() => setCollections([]));

  useEffect(() => { refresh(); }, []);
  useEffect(() => {
    if (!activeCollection && collections[0]) onActiveChange(collections[0].name);
  }, [collections]);

  const selected = activeCollection ?? (collections[0]?.name ?? "");

  const upload = async () => {
    if (!files || files.length === 0) return;
    const destination = uploadMode === "existing" ? selected : sanitize(newName);
    const form = new FormData();
    Array.from(files).forEach((f) => form.append("files", f, f.name));
    const r = await fetch(`/api/upload/${destination}`, { method: "POST", body: form });
    if (r.ok) {
      await refresh();
      onActiveChange(destination);
      setFiles(null);
      alert("Upload finished.");
    } else {
      alert("Upload failed.");
    }
  };

  const remove = async () => {
    if (!selected) return;
    if (!confirm(`Delete collection '${selected}'? This cannot be undone.`)) return;
    const r = await fetch(`/api/collections/${selected}`, { method: "DELETE" });
    if (r.ok) {
      await refresh();
      onActiveChange(collections.filter(c => c.name !== selected)[0]?.name ?? "");
    } else {
      alert("Delete failed.");
    }
  };

  return (
    <div className="card">
      <div className="card-header flex items-center gap-2"><Database size={16}/> Collection Management</div>
      <div className="card-body space-y-4">
        {names.length ? (
          <>
            <label className="text-xs uppercase text-muted">Select Active Collection</label>
            <select
              className="select"
              value={selected}
              onChange={(e) => onActiveChange(e.target.value)}
            >
              {names.map((n) => <option key={n}>{n}</option>)}
            </select>

            <div className="badge">
              Active: <span className="ml-1 font-mono">{selected}</span>
            </div>

            <div className="flex items-center gap-2">
              <button className="btn" onClick={refresh}><RefreshCw size={14} className="mr-2"/>Refresh</button>
              <button className="btn bg-red-600 border-red-600 hover:bg-red-500" onClick={remove}>
                <Trash2 size={14} className="mr-2"/>Delete
              </button>
            </div>
          </>
        ) : (
          <p className="text-sm text-muted">No collections found.</p>
        )}

        <div className="border-t border-zinc-800/60 pt-4 space-y-3">
          <h4 className="text-sm font-semibold">Upload Documents</h4>
          <input
            type="file"
            multiple
            onChange={(e) => setFiles(e.target.files)}
            className="block text-sm file:mr-4 file:py-2 file:px-3 file:rounded-xl file:border-0 file:bg-zinc-800
                       file:text-white hover:file:bg-zinc-700"
          />
          <div className="flex items-center gap-4 text-sm">
            <label className="flex items-center gap-2">
              <input type="radio" checked={uploadMode==="existing"} onChange={()=>setUploadMode("existing")}/> Existing
            </label>
            <label className="flex items-center gap-2">
              <input type="radio" checked={uploadMode==="new"} onChange={()=>setUploadMode("new")}/> Create new
            </label>
          </div>
          {uploadMode==="existing" ? (
            <select className="select" value={selected} onChange={(e)=>onActiveChange(e.target.value)}>
              {names.map((n) => <option key={n}>{n}</option>)}
            </select>
          ) : (
            <input className="input" value={newName} onChange={(e)=>setNewName(e.target.value)} placeholder="new-collection-name"/>
          )}
          <button className="btn btn-primary w-full" onClick={upload}>
            <Upload size={14} className="mr-2"/>Upload Files
          </button>
          <p className="text-xs text-muted">Supported: PDF, DOCX, DOC, TXT, MD</p>
        </div>
      </div>
    </div>
  );
}

function sanitize(name: string) {
  const clean = (name || "").trim().replace(/[^a-zA-Z0-9._-]/g, "_").replace(/^_+|_+$/g, "");
  return clean.length >= 3 ? clean : `${clean}_db`;
}
