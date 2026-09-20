import React, { useState } from 'react';

/**
 * QuestionForm component.
 *
 * @param {object} props
 * @param {boolean} props.isQuerying - Query execution state
 * @param {boolean} props.hasIndexedDoc - Whether a document has been indexed
 * @param {function(string): void} props.onSubmit - Trigger query submission
 */
export function QuestionForm({ isQuerying, hasIndexedDoc, onSubmit }) {
  const [question, setQuestion] = useState('');

  const trimmed = question.trim();
  const isValid = trimmed.length > 0;

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!isValid || isQuerying) return;
    onSubmit(trimmed);
  };

  const handleKeyDown = (e) => {
    // Submit on Enter without Shift
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (isValid && !isQuerying) {
        onSubmit(trimmed);
      }
    }
  };

  return (
    <section className="card question-section" aria-labelledby="query-heading">
      <div className="section-header">
        <h2 id="query-heading" className="section-title">
          Ask a Question
        </h2>
      </div>

      {!hasIndexedDoc && (
        <div className="notice-banner" role="note">
          <span className="notice-icon" aria-hidden="true">💡</span>
          <span>
            Please upload and index a document first so the RAG pipeline has context to retrieve and answer from.
          </span>
        </div>
      )}

      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="question-input" className="form-label">
            Question
          </label>
          <textarea
            id="question-input"
            className="textarea-input"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="e.g., What are the key findings or takeaways described in the document?"
            rows={3}
            disabled={isQuerying}
            required
            aria-describedby="question-hint"
          />
          <span id="question-hint" className="form-hint">
            Press <strong>Enter</strong> to ask, or <strong>Shift + Enter</strong> for a new line.
          </span>
        </div>

        <div className="form-actions">
          <button
            type="submit"
            className="btn btn-primary"
            disabled={!isValid || isQuerying}
          >
            {isQuerying ? (
              <>
                <span className="btn-spinner" aria-hidden="true"></span>
                <span>Generating answer...</span>
              </>
            ) : (
              'Ask Question'
            )}
          </button>
        </div>
      </form>
    </section>
  );
}
