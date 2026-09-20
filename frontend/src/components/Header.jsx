import React from 'react';

/**
 * Header component for MM-RAG UI.
 *
 * @param {object} props
 * @param {{ status: string, environment?: string } | null} props.health - Backend health status
 * @param {boolean} props.isCheckingHealth - Whether health check is in progress
 */
export function Header({ health, isCheckingHealth }) {
  const isHealthy = health?.status === 'healthy';

  return (
    <header className="header">
      <div className="header-brand">
        <div className="header-logo-badge">MM</div>
        <div>
          <h1 className="header-title">MM-RAG</h1>
          <p className="header-subtitle">
            Multi-Modal Multi-Document RAG System
          </p>
        </div>
      </div>
      <div className="header-status">
        <span
          className={`status-pill ${
            isCheckingHealth ? 'status-checking' : isHealthy ? 'status-online' : 'status-offline'
          }`}
          title={
            isCheckingHealth
              ? 'Checking backend...'
              : isHealthy
              ? `Backend connected (${health.environment || 'online'})`
              : 'Backend unreachable'
          }
        >
          <span className="status-dot" aria-hidden="true"></span>
          {isCheckingHealth
            ? 'Checking Backend'
            : isHealthy
            ? 'Backend Online'
            : 'Backend Offline'}
        </span>
      </div>
    </header>
  );
}
