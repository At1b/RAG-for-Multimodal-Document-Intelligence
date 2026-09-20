import React, { useState, useEffect, useCallback } from 'react';
import { Header } from './components/Header';
import { DocumentUpload } from './components/DocumentUpload';
import { QuestionForm } from './components/QuestionForm';
import { AnswerDisplay } from './components/AnswerDisplay';
import { checkHealth, uploadDocument, submitQuery } from './api/client';
import './App.css';

export default function App() {
  // Backend health state
  const [health, setHealth] = useState(null);
  const [isCheckingHealth, setIsCheckingHealth] = useState(true);

  // Document upload state
  const [indexedDoc, setIndexedDoc] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);

  // Query and answer state
  const [isQuerying, setIsQuerying] = useState(false);
  const [queryResult, setQueryResult] = useState(null);
  const [queryError, setQueryError] = useState(null);
  const [lastQuestion, setLastQuestion] = useState(null);

  // Check backend health on initial load
  useEffect(() => {
    let isMounted = true;
    async function verifyBackend() {
      setIsCheckingHealth(true);
      try {
        const data = await checkHealth();
        if (isMounted) {
          setHealth(data);
        }
      } catch (err) {
        if (isMounted) {
          setHealth({ status: 'unreachable', error: err.message });
        }
      } finally {
        if (isMounted) {
          setIsCheckingHealth(false);
        }
      }
    }
    verifyBackend();
    return () => {
      isMounted = false;
    };
  }, []);

  // Handle document upload
  const handleUpload = useCallback(async (file) => {
    setIsUploading(true);
    setUploadError(null);
    try {
      const docResult = await uploadDocument(file);
      setIndexedDoc(docResult);
      // Reset previous answer when new document is uploaded
      setQueryResult(null);
      setQueryError(null);
      setLastQuestion(null);
    } catch (err) {
      setUploadError(err.message || 'Failed to upload and index document.');
    } finally {
      setIsUploading(false);
    }
  }, []);

  // Handle question submission
  const handleQuery = useCallback(async (question) => {
    setIsQuerying(true);
    setQueryError(null);
    setLastQuestion(question);
    try {
      const res = await submitQuery(question);
      setQueryResult(res);
    } catch (err) {
      setQueryResult(null);
      setQueryError(err);
    } finally {
      setIsQuerying(false);
    }
  }, []);

  return (
    <div className="app-container">
      <Header health={health} isCheckingHealth={isCheckingHealth} />

      <main className="main-content">
        <DocumentUpload
          isUploading={isUploading}
          indexedDoc={indexedDoc}
          uploadError={uploadError}
          onUpload={handleUpload}
          onClearError={() => setUploadError(null)}
        />

        <QuestionForm
          isQuerying={isQuerying}
          hasIndexedDoc={Boolean(indexedDoc)}
          onSubmit={handleQuery}
        />

        <AnswerDisplay
          isQuerying={isQuerying}
          queryResult={queryResult}
          queryError={queryError}
          lastQuestion={lastQuestion}
        />
      </main>

      <footer className="footer">
        <p className="footer-text">
          MM-RAG Phase 6 MVP &bull; End-to-End Semantic Retrieval &amp; Generation
        </p>
      </footer>
    </div>
  );
}
