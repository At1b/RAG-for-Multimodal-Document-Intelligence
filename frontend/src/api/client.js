/**
 * MM-RAG API Client — Phase 6 Baseline RAG.
 *
 * Dedicated client for backend communication with /health,
 * /documents/upload, and /query.
 */

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
).replace(/\/+$/, '');

/**
 * Standard API error wrapper containing status and server detail.
 */
export class ApiError extends Error {
  constructor(message, status = 0, detail = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

/**
 * Specialized error for HTTP 404 when retrieval relevance gate filters
 * out chunks and no relevant document context exists to answer the question.
 */
export class InsufficientContextError extends ApiError {
  constructor(detail) {
    const msg =
      typeof detail === 'string' && detail.trim()
        ? detail
        : 'The indexed document does not contain enough relevant information to answer this question.';
    super(msg, 404, detail);
    this.name = 'InsufficientContextError';
  }
}

/**
 * Parse human-readable error detail from a non-2xx Response.
 */
async function parseErrorDetail(response) {
  try {
    const data = await response.json();
    if (data && data.detail) {
      if (Array.isArray(data.detail)) {
        return data.detail.map((item) => item.msg || JSON.stringify(item)).join('; ');
      }
      if (typeof data.detail === 'string') {
        return data.detail;
      }
      return JSON.stringify(data.detail);
    }
  } catch {
    // Response body is not JSON or empty
  }
  return response.statusText || `Request failed with status ${response.status}`;
}

/**
 * Check backend health status.
 *
 * @returns {Promise<{ status: string, environment: string }>}
 */
export async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) {
      const detail = await parseErrorDetail(response);
      throw new ApiError(`Health check failed (${response.status})`, response.status, detail);
    }
    return await response.json();
  } catch (err) {
    if (err instanceof ApiError) {
      throw err;
    }
    throw new ApiError(
      `Cannot connect to backend server at ${API_BASE_URL}. Ensure it is running.`,
      0
    );
  }
}

/**
 * Supported file extensions for document ingestion in Phase 6.
 */
export const SUPPORTED_EXTENSIONS = ['.pdf', '.docx'];

/**
 * Validate a file before attempting upload.
 *
 * @param {File} file
 * @returns {string | null} Error message if invalid, null if valid.
 */
export function validateDocumentFile(file) {
  if (!file) {
    return 'Please select a file to upload.';
  }
  const name = file.name || '';
  const lower = name.toLowerCase();
  const isSupported = SUPPORTED_EXTENSIONS.some((ext) => lower.endsWith(ext));
  if (!isSupported) {
    return 'Unsupported file format. Please upload a .pdf or .docx document.';
  }
  return null;
}

/**
 * Upload and index a document via POST /documents/upload.
 *
 * @param {File} file - PDF or DOCX file object.
 * @returns {Promise<{ document_id: string, document_name: string, num_pages: number, num_chunks: number }>}
 */
export async function uploadDocument(file) {
  const clientValidation = validateDocumentFile(file);
  if (clientValidation) {
    throw new ApiError(clientValidation, 415);
  }

  const formData = new FormData();
  formData.append('file', file);

  let response;
  try {
    response = await fetch(`${API_BASE_URL}/documents/upload`, {
      method: 'POST',
      body: formData,
    });
  } catch {
    throw new ApiError(
      `Cannot connect to backend server at ${API_BASE_URL}. Ensure it is running.`,
      0
    );
  }

  if (!response.ok) {
    const detail = await parseErrorDetail(response);
    throw new ApiError(detail, response.status, detail);
  }

  return await response.json();
}

/**
 * Submit a question to the RAG pipeline via POST /query.
 *
 * @param {string} question - Question string.
 * @param {number | null} [topK=null] - Optional top_k override.
 * @returns {Promise<{ answer: string, model_name: string, num_chunks_retrieved: number }>}
 */
export async function submitQuery(question, topK = null) {
  if (!question || !question.trim()) {
    throw new ApiError('Question cannot be empty.', 400);
  }

  const payload = { question: question.trim() };
  if (typeof topK === 'number' && topK > 0) {
    payload.top_k = topK;
  }

  let response;
  try {
    response = await fetch(`${API_BASE_URL}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });
  } catch {
    throw new ApiError(
      `Cannot connect to backend server at ${API_BASE_URL}. Ensure it is running.`,
      0
    );
  }

  if (response.status === 404) {
    const detail = await parseErrorDetail(response);
    throw new InsufficientContextError(detail);
  }

  if (!response.ok) {
    const detail = await parseErrorDetail(response);
    throw new ApiError(detail, response.status, detail);
  }

  return await response.json();
}
