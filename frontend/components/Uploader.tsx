"use client";
import { useMemo, useState } from "react";
import { uploadToCollection } from "@/lib/api";
import { sanitizeCollectionNameFrontend } from "@/utils/sanitize";

export default function Uploader({
  currentCollections,
  onUploaded
}: {
  currentCollections: string[];
  onUploaded: () => void;
}) {
  const [files, setFiles] = useState<File[]>([]);
  const [mode, setMode] = useState<"existing"|"new">("existing");
  const [existing, setExisting] = useState<string>(currentCollections[0] || "");
  const [newName, setNewName] = useState<string>("my-syllabus-collection");
  const sanitized = useMemo(() => sanitizeCollectionNameFrontend(newName), [newName]);

  return (
    <div className="space-y-3">
      <h3 className="font-semibold">📁 Upload Documents</h3>

      <input
        type="file"
        multiple
        accept=".pdf,.doc,.docx,.txt,.md"
        onChange={e => setFiles(Array.from(e.target.files || []))}
        className="block w-full text-sm"
      />

      <div className="flex gap-4 text-sm">
        <label>
          <input type="radio" checked={mode==="existing"} onChange={() => setMode("existing")} className="mr-2" />
          Existing collection
        </label>
        <label>
          <input type="radio" checked={mode==="new"} onChange={() => setMode("new")} className="mr-2" />
          Create new collection
        </label>
      </div>

      {mode === "existing" ? (
        <select className="border rounded px-2 py-1" value={existing} onChange={e=>setExisting(e.target.value)}>
          {currentCollections.map(n => <option key={n} value={n}>{n}</option>)}
        </select>
      ) : (
        <div className="text-sm">
          <input
            className="border rounded px-2 py-1 w-full"
            value={newName}
            onChange={e=>setNewName(e.target.value)}
            placeholder="my-syllabus-collection"
          />
          <div className="text-gray-600">Sanitized name: <code>{sanitized}</code></div>
          {currentCollections.includes(sanitized) && (
            <div className="text-red-600">A collection with this name already exists. Pick another.</div>
          )}
        </div>
      )}

      {files.length > 0 && (
        <div className="text-sm">
          <b>{files.length}</b> file(s) selected. Total size:{" "}
          { (files.reduce((s,f)=>s+f.size,0)/1024/1024).toFixed(2) } MB
        </div>
      )}

      <button
        className="px-3 py-1 border rounded bg-blue-600 text-white disabled:opacity-50"
        disabled={!files.length || (mode==="new" && (currentCollections.includes(sanitized) || !sanitized))}
        onClick={async()=>{
          const dest = mode === "existing" ? existing : sanitized;
          try {
            const res = await uploadToCollection(dest, files);
            alert("Upload completed successfully.");
            onUploaded();
            setFiles([]);
          } catch (e:any) {
            alert(`Upload failed: ${e?.message || e}`);
          }
        }}
      >
        🚀 Upload Files
      </button>
    </div>
  );
}
