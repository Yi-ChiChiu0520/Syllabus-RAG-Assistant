"use client";
import useSWR from "swr";
import { deleteCollection, listCollections } from "@/lib/api";
import { useEffect, useState } from "react";
import { sanitizeCollectionNameFrontend } from "@/utils/sanitize";
import Uploader from "./Uploader";

const fetcher = async () => await listCollections();

export default function CollectionsPanel({
  selected,
  onSelect,
  onRefresh
}: {
  selected: string | null;
  onSelect: (name: string) => void;
  onRefresh: () => void;
}) {
  const { data, error, isLoading, mutate } = useSWR("collections", fetcher, { refreshInterval: 30_000 });
  const [confirmDelete, setConfirmDelete] = useState(false);

  useEffect(() => {
    if (data && data.length && !selected) onSelect(data[0].name);
  }, [data, selected, onSelect]);

  if (error) return <div className="text-red-600">Failed to load collections</div>;

  const names = (data || []).map(c => c.name);
  const selectedData = (data || []).find(c => c.name === selected);

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold">📂 Collection Management</h2>

      {isLoading && <div className="text-sm">Loading collections…</div>}
      {!isLoading && !data?.length && (
        <div className="text-sm">⚠️ No collections found. Upload files to create one.</div>
      )}

      {data?.length ? (
        <>
          <div className="space-y-2">
            <label className="text-sm">Select Active Collection</label>
            <select
              className="border rounded px-2 py-1 w-full"
              value={selected || ""}
              onChange={e => onSelect(e.target.value)}
            >
              {names.map(n => <option key={n} value={n}>{n}</option>)}
            </select>
            <div className="text-sm">Active Collection: <code>{selected}</code></div>
            {selectedData && (
              <div className="text-sm bg-gray-50 border rounded p-2">
                📈 Total Documents: {selectedData.document_count}
                {selectedData.source_files && Object.keys(selectedData.source_files).length > 0 ? (
                  <details className="mt-2">
                    <summary className="cursor-pointer">📄 Files in Collection ({Object.keys(selectedData.source_files).length})</summary>
                    <ul className="list-disc ml-6">
                      {Object.entries(selectedData.source_files).sort().map(([file, cnt]) => (
                        <li key={file}><b>{file}</b> — {cnt} chunk(s)</li>
                      ))}
                    </ul>
                  </details>
                ) : (
                  <div className="mt-2">No source file metadata found.</div>
                )}
              </div>
            )}
          </div>

          <Uploader currentCollections={names} onUploaded={() => { mutate(); onRefresh(); }} />

          <div className="flex items-center gap-3">
            <button className="px-3 py-1 border rounded" onClick={() => { mutate(); onRefresh(); }}>
              🔄 Refresh Collections
            </button>
            <label className="text-sm flex items-center gap-2">
              <input type="checkbox" checked={confirmDelete} onChange={e => setConfirmDelete(e.target.checked)} />
              ⚠️ Confirm deletion of '{selected}'
            </label>
            <button
              className="px-3 py-1 border rounded text-red-700"
              onClick={async () => {
                if (!selected) return;
                if (!confirmDelete) { alert("Please confirm first."); return; }
                try {
                  await deleteCollection(selected);
                  setConfirmDelete(false);
                  await mutate();
                  onSelect((data || []).filter(c => c.name !== selected)[0]?.name || null);
                  onRefresh();
                } catch (e:any) {
                  alert(`Delete failed: ${e?.message || e}`);
                }
              }}
            >
              ⚠️ Delete Collection
            </button>
          </div>
        </>
      ) : null}
    </div>
  );
}
