import React from 'react';
import { InsufficientContextError } from '../api/client';

/**
 * AnswerDisplay component.
 *
 * @param {object} props
 * @param {boolean} props.isQuerying - Whether answer generation is running
 * @param {{ answer: string, model_name: string, num_chunks_retrieved: number, sources?: Array<{ document_id: string, document_name: string, page_number: number|null, chunk_id: string, score: number }> } | null} props.queryResult - Generation result
 * @param {Error | null} props.queryError - Query error object if any
 * @param {string | null} props.lastQuestion - The question corresponding to current answer
 */
export function AnswerDisplay({
  isQuerying,
  queryResult,
  queryError,
  lastQuestion,
}) {
  const isInsufficientContext =
    queryError instanceof InsufficientContextError || queryError?.status === 404;

  const sources = queryResult?.sources || [];

  return (
    <section className="card answer-section" aria-labelledby="answer-heading">
      <div className="section-header">
        <h2 id="answer-heading" className="section-title">
          Generated Answer
        </h2>
        {queryResult && (
          <div className="meta-chips">
            <span className="meta-chip" title="Retrieved Context Chunks">
              <span className="meta-chip-label">Retrieved Chunks:</span>{' '}
              <strong>{queryResult.num_chunks_retrieved}</strong>
            </span>
            <span className="meta-chip" title="Model Used">
              <span className="meta-chip-label">Model:</span>{' '}
              <code>{queryResult.model_name}</code>
            </span>
          </div>
        )}
      </div>

      {/* Loading state */}
      {isQuerying && (
        <div className="state-card state-loading" aria-live="polite">
          <div className="spinner-large" aria-hidden="true"></div>
          <div className="state-loading-text">
            <h3>Generating answer...</h3>
            <p>Retrieving relevant document chunks and synthesizing response.</p>
          </div>
        </div>
      )}

      {/* 404 Insufficient Context / Relevance Gate */}
      {!isQuerying && isInsufficientContext && (
        <div className="alert alert-warning" role="alert" aria-live="assertive">
          <div className="alert-header">
            <span className="alert-icon" aria-hidden="true">🔍</span>
            <h3 className="alert-title">Insufficient Relevant Context</h3>
          </div>
          <p className="alert-message">
            {queryError.message ||
              'The indexed documents do not contain enough relevant information to answer this question.'}
          </p>
          <p className="alert-suggestion">
            Try rephrasing your question or ask about topics covered in the uploaded documents.
          </p>
        </div>
      )}

      {/* Generic Error */}
      {!isQuerying && queryError && !isInsufficientContext && (
        <div className="alert alert-error" role="alert" aria-live="assertive">
          <div className="alert-header">
            <span className="alert-icon" aria-hidden="true">⚠️</span>
            <h3 className="alert-title">Query Processing Failed</h3>
          </div>
          <p className="alert-message">{queryError.message}</p>
        </div>
      )}

      {/* Successful Answer */}
      {!isQuerying && queryResult && (
        <div className="answer-content-box" aria-live="polite">
          {lastQuestion && (
            <div className="question-recap">
              <span className="question-recap-label">Q:</span>
              <span className="question-recap-text">{lastQuestion}</span>
            </div>
          )}
          <div className="answer-text">
            {queryResult.answer}
          </div>

          {/* Phase 7: Sources / Citations */}
          {sources.length > 0 && (
            <div className="sources-section" aria-labelledby="sources-heading">
              <h3 id="sources-heading" className="sources-title">
                📄 Sources
              </h3>
              <ul className="sources-list">
                {sources.map((source, idx) => (
                  <li key={source.chunk_id || idx} className="source-item">
                    <span className="source-index">[{idx + 1}]</span>
                    <span className="source-doc-name">{source.document_name}</span>
                    {source.page_number != null && (
                      <span className="source-page">— Page {source.page_number}</span>
                    )}
                    {typeof source.score === 'number' && !isNaN(source.score) && (
                      <span className="source-score" title="Similarity score">
                        ({(source.score * 100).toFixed(1)}% match)
                      </span>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Empty State */}
      {!isQuerying && !queryResult && !queryError && (
        <div className="state-card state-empty">
          <span className="empty-icon" aria-hidden="true">💬</span>
          <p className="empty-title">No question asked yet</p>
          <p className="empty-description">
            Ask a question above to retrieve grounded answers from your indexed documents.
          </p>
        </div>
      )}
    </section>
  );
}
