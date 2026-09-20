import React, { useState, useRef } from 'react';
import { validateDocumentFile } from '../api/client';

/**
 * Format bytes to readable size string.
 */
function formatFileSize(bytes) {
  if (!bytes || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`;
}

/**
 * DocumentUpload component.
 *
 * @param {object} props
 * @param {boolean} props.isUploading - Upload state
 * @param {{ document_id: string, document_name: string, num_pages: number, num_chunks: number } | null} props.indexedDoc - Currently indexed document metadata
 * @param {string | null} props.uploadError - Error message from upload
 * @param {function(File): void} props.onUpload - Trigger upload callback
 * @param {function(): void} props.onClearError - Clear upload error callback
 */
export function DocumentUpload({
  isUploading,
  indexedDoc,
  uploadError,
  onUpload,
  onClearError,
}) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [clientError, setClientError] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileChange = (file) => {
    onClearError();
    if (!file) {
      setSelectedFile(null);
      setClientError(null);
      return;
    }

    const validationError = validateDocumentFile(file);
    if (validationError) {
      setClientError(validationError);
      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      return;
    }

    setClientError(null);
    setSelectedFile(file);
  };

  const handleInputChange = (e) => {
    const file = e.target.files?.[0] || null;
    handleFileChange(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (isUploading) return;
    const file = e.dataTransfer.files?.[0] || null;
    handleFileChange(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    if (!isUploading) {
      setIsDragOver(true);
    }
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!selectedFile || isUploading) return;
    onUpload(selectedFile);
  };

  const errorMessage = clientError || uploadError;

  return (
    <section className="card upload-section" aria-labelledby="upload-heading">
      <div className="section-header">
        <h2 id="upload-heading" className="section-title">
          Document Ingestion
        </h2>
        <span className="section-badge">Phase 6 Baseline</span>
      </div>
      <p className="section-description">
        Upload a document to parse, chunk, embed, and index into the vector store.
      </p>

      <form onSubmit={handleSubmit}>
        <div
          className={`dropzone ${isDragOver ? 'dropzone-active' : ''} ${
            isUploading ? 'dropzone-disabled' : ''
          }`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => !isUploading && fileInputRef.current?.click()}
          role="button"
          tabIndex={isUploading ? -1 : 0}
          onKeyDown={(e) => {
            if ((e.key === 'Enter' || e.key === ' ') && !isUploading) {
              e.preventDefault();
              fileInputRef.current?.click();
            }
          }}
          aria-label="Upload document area. Click or drop a PDF or DOCX file."
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleInputChange}
            accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            disabled={isUploading}
            style={{ display: 'none' }}
            id="document-file-input"
          />

          <div className="dropzone-icon" aria-hidden="true">
            📄
          </div>

          {selectedFile ? (
            <div className="selected-file-info">
              <span className="selected-file-name">{selectedFile.name}</span>
              <span className="selected-file-size">
                ({formatFileSize(selectedFile.size)})
              </span>
              <span className="change-file-hint">Click to change file</span>
            </div>
          ) : (
            <div className="dropzone-instructions">
              <span className="dropzone-primary-text">
                Choose a PDF or DOCX file or drag it here
              </span>
              <span className="dropzone-secondary-text">
                Supported formats: <strong>.pdf</strong>, <strong>.docx</strong> (max 50 MB)
              </span>
            </div>
          )}
        </div>

        {errorMessage && (
          <div className="alert alert-error" role="alert">
            <span className="alert-icon" aria-hidden="true">⚠️</span>
            <div className="alert-content">
              <strong>Upload Error:</strong> {errorMessage}
            </div>
          </div>
        )}

        <div className="form-actions">
          <button
            type="submit"
            className="btn btn-primary"
            disabled={!selectedFile || isUploading}
          >
            {isUploading ? (
              <>
                <span className="btn-spinner" aria-hidden="true"></span>
                <span>Indexing document...</span>
              </>
            ) : (
              'Upload & Index Document'
            )}
          </button>
        </div>
      </form>

      {/* Success state after indexing */}
      {indexedDoc && (
        <div className="indexed-card" aria-live="polite">
          <div className="indexed-header">
            <span className="indexed-badge" aria-hidden="true">✓</span>
            <div className="indexed-title-group">
              <h3 className="indexed-title">Document Indexed Successfully</h3>
              <span className="indexed-doc-name">{indexedDoc.document_name}</span>
            </div>
          </div>
          <div className="indexed-metrics-grid">
            <div className="metric-box">
              <span className="metric-label">Pages</span>
              <span className="metric-value">{indexedDoc.num_pages}</span>
            </div>
            <div className="metric-box">
              <span className="metric-label">Indexed Chunks</span>
              <span className="metric-value">{indexedDoc.num_chunks}</span>
            </div>
            <div className="metric-box metric-box-wide">
              <span className="metric-label">Document ID</span>
              <code className="metric-code" title={indexedDoc.document_id}>
                {indexedDoc.document_id}
              </code>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
