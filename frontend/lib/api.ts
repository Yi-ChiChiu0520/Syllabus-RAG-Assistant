// Fix: Use consistent variable name and provide fallback
export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

console.log("API_BASE_URL", API_BASE_URL);

// Test the connection immediately
fetch(`${API_BASE_URL}/health`, { cache: "no-store" })
  .then(r => r.text())
  .then(t => console.log("health:", t))
  .catch(e => console.error("health error:", e));

export async function health(): Promise<{status: string; ready?: boolean; timestamp?: number}> {
  // Fix: Use API_BASE_URL instead of API_BASE
  const r = await fetch(`${API_BASE_URL}/health`, { cache: "no-store" });
  if (!r.ok) throw new Error("health failed");
  return r.json();
}

export async function listCollections(): Promise<Array<{
  name: string; 
  document_count: number; 
  source_files: Record<string, number>
}>> {
  // Fix: Use API_BASE_URL instead of API_BASE
  const r = await fetch(`${API_BASE_URL}/collections`, { cache: "no-store" });
  if (!r.ok) throw new Error("collections failed");
  return r.json();
}

export async function listModels(): Promise<{
  ollama_models: string[]; 
  openai_available: boolean
}> {
  // Fix: Use API_BASE_URL instead of API_BASE
  const r = await fetch(`${API_BASE_URL}/models`, { cache: "no-store" });
  if (!r.ok) throw new Error("models failed");
  return r.json();
}

export async function deleteCollection(name: string): Promise<void> {
  // Fix: Use API_BASE_URL instead of API_BASE
  const r = await fetch(`${API_BASE_URL}/collections/${encodeURIComponent(name)}`, { 
    method: "DELETE" 
  });
  if (!r.ok) throw new Error(await r.text());
}

export async function uploadToCollection(collection: string, files: File[]): Promise<any> {
  const form = new FormData();
  files.forEach(f => form.append("files", f, f.name));
  
  // Fix: Use API_BASE_URL instead of API_BASE
  const r = await fetch(`${API_BASE_URL}/upload/${encodeURIComponent(collection)}`, { 
    method: "POST", 
    body: form 
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function queryRag(payload: {
  query: string;
  collection_name: string;
  model_type: "openai" | "ollama";
  model_name: string;
}): Promise<any> {
  // Fix: Use API_BASE_URL instead of API_BASE
  const r = await fetch(`${API_BASE_URL}/query`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}