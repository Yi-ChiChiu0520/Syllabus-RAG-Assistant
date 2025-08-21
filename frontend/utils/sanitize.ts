export function sanitizeCollectionNameFrontend(name: string): string {
  let clean = (name || "").trim().replace(/[^a-zA-Z0-9._-]/g, "_").replace(/^_+|_+$/g, "");
  if (clean.length < 3) clean = `${clean}_db`;
  return clean;
}
