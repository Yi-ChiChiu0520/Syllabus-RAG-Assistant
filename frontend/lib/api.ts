// Debug environment variables first
console.log("Environment check:", {
  BACKEND_API_BASE_URL: process.env.BACKEND_API_BASE_URL,
  NEXT_PUBLIC_API_BASE_URL: process.env.NEXT_PUBLIC_API_BASE_URL,
  NODE_ENV: process.env.NODE_ENV
});

// Fix: Prioritize the correct environment variable and remove localhost fallback for production
export const API_BASE_URL = (() => {
  const serverSideUrl = process.env.BACKEND_API_BASE_URL;
  const clientSideUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
  const fallbackUrl = "http://localhost:8000";

  // In production, don't fallback to localhost
  if (process.env.NODE_ENV === 'production') {
    const prodUrl = clientSideUrl || serverSideUrl;
    if (!prodUrl) {
      throw new Error("Production environment requires NEXT_PUBLIC_API_BASE_URL to be set");
    }
    return prodUrl.replace(/\/+$/, "");
  }

  // In development, use env vars first, then fallback to localhost
  const devUrl = clientSideUrl || serverSideUrl || fallbackUrl;
  return devUrl.replace(/\/+$/, "");
})();

console.log("Using API_BASE_URL:", API_BASE_URL);

// Test the connection immediately with better error handling
const testConnection = async () => {
  try {
    console.log(`Testing connection to: ${API_BASE_URL}/health`);
    const response = await fetch(`${API_BASE_URL}/health`, {
      cache: "no-store",
      signal: AbortSignal.timeout(10000) // 10 second timeout
    });

    if (response.ok) {
      const data = await response.text();
      console.log("✅ Backend connection successful:", data);
    } else {
      console.error(`❌ Backend responded with status ${response.status}:`, await response.text());
    }
  } catch (error) {
    console.error("❌ Backend connection failed:", error);

    // Additional debugging for CORS/network issues
    if (error instanceof TypeError && error.message.includes('fetch')) {
      console.error("This might be a CORS issue or the backend is not accessible");
    }
  }
};

// Run connection test
testConnection();

// Helper function with better error handling and timeouts
async function apiCall<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

  try {
    const response = await fetch(url, {
      cache: "no-store",
      signal: AbortSignal.timeout(30000), // 30 second timeout
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      }
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status} for ${endpoint}: ${errorText}`);
    }

    return response.json();
  } catch (error) {
    console.error(`API call failed for ${endpoint}:`, error);

    // Provide more specific error messages
    if (error instanceof TypeError && error.message.includes('fetch')) {
      throw new Error(`Network error: Unable to connect to ${url}. Check if the backend is running and accessible.`);
    }

    throw error;
  }
}

export async function health(): Promise<{status: string; ready?: boolean; timestamp?: number}> {
  return apiCall('/health');
}

export async function listCollections(): Promise<Array<{
  name: string;
  document_count: number;
  source_files: Record<string, number>
}>> {
  return apiCall('/collections');
}

export async function listModels(): Promise<{
  ollama_models: string[];
  openai_available: boolean
}> {
  return apiCall('/models');
}

export async function deleteCollection(name: string): Promise<void> {
  await apiCall(`/collections/${encodeURIComponent(name)}`, {
    method: "DELETE"
  });
}

export async function uploadToCollection(collection: string, files: File[]): Promise<any> {
  const form = new FormData();
  files.forEach(f => form.append("files", f, f.name));

  const url = `${API_BASE_URL}/upload/${encodeURIComponent(collection)}`;

  try {
    const response = await fetch(url, {
      method: "POST",
      body: form,
      signal: AbortSignal.timeout(300000) // 5 minute timeout for uploads
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Upload failed: ${errorText}`);
    }

    return response.json();
  } catch (error) {
    console.error('Upload error:', error);
    throw error;
  }
}

export async function queryRag(payload: {
  query: string;
  collection_name: string;
  model_type: "openai" | "ollama";
  model_name: string;
}): Promise<any> {
  return apiCall('/query', {
    method: "POST",
    body: JSON.stringify(payload)
  });
}