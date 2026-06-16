"""HTML report generation for lower trace."""

from __future__ import annotations

import os

from .core import LowerRecord, STATUS_COMPLETED, STATUS_FAILED, STATUS_SKIPPED, STATUS_CODEGEN
from .diff import _esc, _make_diff_html


_CSS = """
:root {
    --bg: #f5f7fa;
    --bg-elevated: #ffffff;
    --bg-inset: #e2e8f0;
    --bg-hover: #f1f5f9;
    --bg-active: #eff6ff;
    --bg-code: #f6f8fa;
    --bg-header-hover: #f8fafc;
    --bg-btn-hover: #f3f4f6;
    --border: #e2e8f0;
    --border-strong: #cbd5e1;
    --border-input: #d0d7de;
    --text: #1e293b;
    --text-secondary: #334155;
    --text-muted: #64748b;
    --text-faint: #8c959f;
    --text-fainter: #94a3b8;
    --text-dim: #475569;
    --text-diff: #24292f;
    --text-dot-noop: #d1d5db;
    --accent: #2563eb;
    --accent-blue: #3b82f6;
    --green: #22c55e;
    --green-text: #166534;
    --green-bg: #dcfce7;
    --green-dark: #1a7f37;
    --green-light: #abf2ca;
    --green-bg-light: #dafbe1;
    --green-word: #acf2bd;
    --green-text-dark: #116329;
    --red: #dc2626;
    --red-text: #991b1b;
    --red-text-dark: #82071e;
    --red-bg: #ffebe9;
    --red-bg-light: #ffd7d5;
    --red-bg-bright: #fecaca;
    --red-dark: #cf222e;
    --red-word: #ffcecb;
    --red-border: #fca5a5;
    --red-accent: #ef4444;
    --red-bg-lighter: #fef2f2;
    --amber: #f59e0b;
    --failed-bg: #fffbfb;
    --skipped-bg: #fffdf5;
    --yellow-bg: #fff8c5;
    --purple-bg: #f0f0ff;
    --purple-bg-light: #e8e8f8;
    --purple-text: #4040a0;
    --purple-text-light: #6060b0;
    --purple-word: #d8d8f0;
    --align-left-from: #0f172a;
    --align-left-to: #1e293b;
    --align-pending-bg: #fef3c7;
    --align-pending-text: #664d03;
    --align-right-bg: #cfe2ff;
    --align-right-text: #084298;
    --align-locked-bg: #dbeafe;
    --blue-bg-light: #ddf4ff;
    --blue-bg-hover: #c8e9ff;
    --toast-bg: #1f2937;
}

[data-theme="dark"] {
    --bg: #1e1e2e;
    --bg-elevated: #313244;
    --bg-inset: #181825;
    --bg-hover: #45475a;
    --bg-active: #1e3a5f;
    --bg-code: #262636;
    --bg-header-hover: #45475a;
    --bg-btn-hover: #585b70;
    --border: #45475a;
    --border-strong: #585b70;
    --border-input: #585b70;
    --text: #cdd6f4;
    --text-secondary: #bac2de;
    --text-muted: #a6adc8;
    --text-faint: #6c7086;
    --text-fainter: #585b70;
    --text-dim: #a6adc8;
    --text-diff: #cdd6f4;
    --text-dot-noop: #585b70;
    --accent: #89b4fa;
    --accent-blue: #89b4fa;
    --green: #a6e3a1;
    --green-text: #a6e3a1;
    --green-bg: rgba(166, 227, 161, 0.12);
    --green-dark: #a6e3a1;
    --green-light: #2d4a2d;
    --green-bg-light: #2d4a2d;
    --green-word: #3a6a3a;
    --green-text-dark: #a6e3a1;
    --red: #f38ba8;
    --red-text: #f38ba8;
    --red-text-dark: #f38ba8;
    --red-bg: #2e1a1a;
    --red-bg-light: #2e1a1a;
    --red-bg-bright: #4a2d2d;
    --red-dark: #f38ba8;
    --red-word: #5a3030;
    --red-border: #5a3030;
    --red-accent: #f38ba8;
    --red-bg-lighter: #2e1a1a;
    --amber: #fab387;
    --failed-bg: #2a1a1e;
    --skipped-bg: #2a2418;
    --yellow-bg: #3e3e5e;
    --purple-bg: #262636;
    --purple-bg-light: #313244;
    --purple-text: #b4befe;
    --purple-text-light: #7f849c;
    --purple-word: #45475a;
    --align-left-from: #1a1a2e;
    --align-left-to: #1e1e2e;
    --align-pending-bg: #3e3a2e;
    --align-pending-text: #f9e2af;
    --align-right-bg: #1e3a5f;
    --align-right-text: #89dceb;
    --align-locked-bg: #1e3a5f;
    --blue-bg-light: #1e3a5f;
    --blue-bg-hover: #2e4a6f;
    --toast-bg: #cdd6f4;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.header {
    background: linear-gradient(135deg, var(--align-left-from), var(--align-left-to));
    color: white;
    padding: 12px 20px;
    flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    position: relative;
}
.header h1 { font-size: 17px; font-weight: 600; }
.header .sub { font-size: 12px; opacity: 0.6; margin-top: 2px; }

.theme-btn {
    position: absolute;
    right: 20px;
    top: 12px;
    padding: 4px 12px;
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 4px;
    background: rgba(255,255,255,0.1);
    color: white;
    cursor: pointer;
    font-size: 12px;
    font-family: inherit;
    transition: background 0.15s;
    z-index: 10;
}
.theme-btn:hover { background: rgba(255,255,255,0.2); }

.phase-tabs {
    display: flex;
    background: var(--bg-inset);
    border-bottom: 1px solid var(--border-strong);
    flex-shrink: 0;
}
.phase-tab {
    padding: 8px 20px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-muted);
    border-bottom: 3px solid transparent;
    transition: all 0.15s;
    user-select: none;
}
.phase-tab:hover { background: var(--bg-hover); color: var(--text); }
.phase-tab.active {
    color: var(--accent);
    border-bottom-color: var(--accent);
    background: var(--bg);
}

.summary-bar {
    background: var(--bg-elevated);
    padding: 8px 20px;
    border-bottom: 1px solid var(--border);
    font-size: 13px;
    flex-shrink: 0;
}
.summary-bar .badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 10px;
    margin-right: 8px;
    font-weight: 600;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
    border: 2px solid transparent;
    user-select: none;
}
.summary-bar .badge:hover { filter: brightness(0.92); }
.summary-bar .badge.active { border-color: var(--text); box-shadow: 0 0 0 1px var(--text); }
.summary-bar .badge.dimmed { opacity: 0.35; }
.badge-total   { background: var(--bg-inset); color: var(--text-dim); }
.badge-changed { background: var(--green-bg); color: var(--green-text); }
.badge-noop    { background: var(--bg-hover); color: var(--text-fainter); }
.badge-failed  {
    background: var(--red); color: #fff;
    font-weight: 700; font-size: 12px;
    padding: 3px 12px;
    animation: badgePulse 1.5s ease-in-out infinite;
}
.badge-skipped {
    background: var(--amber); color: #fff;
    font-weight: 700; font-size: 12px;
    padding: 3px 12px;
}
.badge-codegen {
    background: var(--purple-bg); color: var(--purple-text);
    font-weight: 700; font-size: 12px;
    padding: 3px 12px;
}
@keyframes badgePulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.4); }
    50% { box-shadow: 0 0 0 4px rgba(220, 38, 38, 0); }
}

.main {
    display: flex;
    flex: 1;
    overflow: hidden;
    position: relative;
}

.sidebar {
    width: 270px;
    min-width: 0;
    background: var(--bg-elevated);
    border-right: 1px solid var(--border);
    overflow-y: auto;
    overflow-x: hidden;
    padding: 8px 0;
    flex-shrink: 0;
    transition: width 0.2s ease;
    position: relative;
}
.sidebar.collapsed {
    width: 0 !important;
    padding: 0;
    border-right: none;
}
.sidebar .section-title {
    padding: 8px 14px 4px;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-fainter);
    font-weight: 700;
    white-space: nowrap;
}

.sidebar-resize {
    width: 5px;
    cursor: col-resize;
    background: transparent;
    flex-shrink: 0;
    position: relative;
    z-index: 20;
    margin-left: -3px;
    margin-right: -2px;
}
.sidebar-resize:hover,
.sidebar-resize.active { background: var(--accent-blue); }

.sidebar-toggle-btn {
    position: absolute;
    top: 8px;
    z-index: 30;
    width: 28px;
    height: 28px;
    border: 1px solid var(--border-input);
    border-radius: 6px;
    background: var(--bg-elevated);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0;
    transition: background 0.15s, box-shadow 0.15s;
    user-select: none;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.sidebar-toggle-btn:hover { background: var(--bg-hover); box-shadow: 0 1px 4px rgba(0,0,0,0.12); }

.sidebar-toggle-btn .chevron {
    display: block;
    width: 7px;
    height: 7px;
    border-right: 2px solid var(--text-muted);
    border-bottom: 2px solid var(--text-muted);
    transition: transform 0.25s ease;
}
.sidebar-toggle-btn:hover .chevron { border-color: var(--text); }

.sidebar-toggle-btn.inside {
    left: auto;
    right: 10px;
    box-shadow: none;
    border-color: var(--border);
}
.sidebar-toggle-btn.inside .chevron {
    transform: rotate(135deg);
    margin-left: 2px;
}

#sidebar-open-btn {
    left: 4px;
}
#sidebar-open-btn .chevron {
    transform: rotate(-45deg);
    margin-right: 2px;
}

.pass-link {
    display: flex;
    align-items: center;
    padding: 5px 14px;
    font-size: 12.5px;
    cursor: pointer;
    color: var(--text-secondary);
    text-decoration: none;
    transition: background 0.1s, opacity 0.15s;
    gap: 7px;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
}
.pass-link:hover { background: var(--bg-hover); }
.pass-link.active { background: var(--bg-active); color: var(--accent); }
.pass-link.filtered-out { display: none; }

.pass-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.pass-dot.changed { background: var(--green); }
.pass-dot.noop    { background: var(--text-dot-noop); }
.pass-dot.failed {
    background: var(--red);
    width: 10px; height: 10px;
    box-shadow: 0 0 0 2px var(--red-bg-bright), 0 0 6px rgba(220,38,38,0.5);
    animation: dotPulse 1.5s ease-in-out infinite;
}
.pass-dot.skipped {
    background: transparent;
    border: 2px solid var(--amber);
    width: 10px; height: 10px;
}
.pass-dot.codegen {
    background: var(--accent-blue);
    width: 10px; height: 10px;
    box-shadow: 0 0 0 2px rgba(37,99,235,0.2), 0 0 6px rgba(37,99,235,0.4);
}
@keyframes dotPulse {
    0%, 100% { box-shadow: 0 0 0 2px var(--red-bg-bright), 0 0 6px rgba(220,38,38,0.5); }
    50% { box-shadow: 0 0 0 4px var(--red-bg-bright), 0 0 10px rgba(220,38,38,0.3); }
}

.pass-label {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    flex: 1;
}

.pass-idx {
    font-size: 10px;
    color: var(--text-fainter);
    flex-shrink: 0;
    width: 18px;
    text-align: right;
}

.pass-stats {
    font-size: 10px;
    flex-shrink: 0;
    white-space: nowrap;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
}
.pass-stats .st-add { color: var(--green-dark); }
.pass-stats .st-del { color: var(--red-dark); }

.content {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
}

.pass-section {
    display: none;
    background: var(--bg-elevated);
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.05);
    padding: 20px;
    margin-bottom: 16px;
}
.pass-section.active { display: block; }
.pass-section.failed-section {
    border-left: 4px solid var(--red);
    background: var(--failed-bg);
}
.pass-section.skipped-section {
    border-left: 4px solid var(--amber);
    background: var(--skipped-bg);
    opacity: 0.85;
}

.pass-section.collapsed > *:not(.pass-header) { display: none; }

.pass-header {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--bg-hover);
    border-radius: 4px;
    transition: background 0.1s;
    position: sticky;
    top: 0;
    z-index: 10;
    background: var(--bg-elevated);
}
.pass-header.collapsible { cursor: pointer; user-select: none; }
.pass-header.collapsible:hover { background: var(--bg-header-hover); }

.pass-toggle {
    width: 16px;
    flex-shrink: 0;
    font-size: 10px;
    color: var(--text-fainter);
    text-align: center;
}
.pass-section:not(.collapsed) > .pass-header .pass-toggle::before { content: '\\25BC'; }
.pass-section.collapsed > .pass-header .pass-toggle::before { content: '\\25B6'; }

.pass-header h2 { font-size: 15px; font-weight: 600; }
.pass-header .status {
    font-size: 11px;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 10px;
    letter-spacing: 0.03em;
    margin-left: auto;
}
.status-changed { background: var(--green-bg); color: var(--green-text); }
.status-noop    { background: var(--bg-hover); color: var(--text-fainter); }
.status-failed {
    background: var(--red); color: #fff;
    font-size: 12px; padding: 3px 14px;
    border-radius: 10px;
    animation: badgePulse 1.5s ease-in-out infinite;
}
.status-skipped {
    background: var(--amber); color: #fff;
    font-size: 12px; padding: 3px 14px;
    border-radius: 10px;
}
.status-codegen {
    background: var(--purple-bg); color: var(--purple-text);
    font-size: 11px; padding: 2px 10px;
    border-radius: 10px;
    border: 1px solid var(--purple-text-light, var(--accent-blue));
}

.pass-section.codegen-section {
    border-left: 4px solid var(--purple-text, var(--accent-blue));
}

.codegen-lang-label {
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 4px;
    margin-bottom: 6px;
}
.codegen-lang-label.tir  { background: var(--purple-bg-light, var(--bg-inset)); color: var(--purple-text, var(--text-dim)); }
.codegen-lang-label.cpp  { background: var(--bg-active, var(--bg-inset)); color: var(--accent-blue, var(--text-dim)); }

.codegen-lang-bar {
    display: flex;
    gap: 16px;
    align-items: center;
    margin-bottom: 8px;
    padding: 4px 8px;
    background: var(--bg-inset, var(--bg-elevated));
    border-radius: 4px;
    font-size: 11px;
}
.codegen-lang-bar::before {
    content: "▸";
    color: var(--text-faint);
    margin-right: 4px;
    font-size: 10px;
}

.error-box {
    background: var(--red-bg-lighter);
    border: 1px solid var(--red-border);
    border-left: 4px solid var(--red-accent);
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 14px;
    font-size: 13px;
    color: var(--red-text);
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    line-height: 1.5;
    word-break: break-word;
}
.error-box .error-label {
    font-weight: 700;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
    color: var(--red);
}

.noop-msg {
    color: var(--text-fainter);
    font-size: 13px;
    text-align: center;
    padding: 16px;
}

.ir-toggle {
    display: block;
    margin: 10px auto 0;
    padding: 5px 18px;
    background: var(--bg-hover);
    border: 1px solid var(--border);
    border-radius: 5px;
    cursor: pointer;
    font-size: 12px;
    color: var(--text-muted);
    transition: background 0.1s;
}
.ir-toggle:hover { background: var(--border); }

.ir-block {
    display: none;
    margin-top: 10px;
    max-height: 500px;
    overflow: auto;
}
.ir-block.show { display: block; }

.ir-block pre {
    background: var(--bg-elevated);
    color: var(--text-diff);
    padding: 14px;
    border-radius: 6px;
    border: 1px solid var(--border-input);
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    line-height: 20px;
    white-space: pre;
    tab-size: 2;
}
.ir-block .ir-line {
    display: block;
}
.ir-block .ir-line.hl {
    background: var(--yellow-bg);
}
.ir-block .ir-ln {
    display: inline-block;
    width: 50px;
    text-align: right;
    padding-right: 12px;
    color: var(--text-faint);
    user-select: none;
    cursor: pointer;
}
.ir-block .ir-ln:hover {
    text-decoration: underline;
}

.diff-table-wrap {
    overflow-x: auto;
    border-radius: 6px;
    border: 1px solid var(--border-input);
    background: var(--bg-elevated);
}
.diff-table-wrap table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 12px;
    line-height: 20px;
    table-layout: fixed;
}
.diff-table-wrap td {
    padding: 0 10px;
    white-space: pre-wrap;
    word-break: break-all;
    vertical-align: top;
}

.diff-table-wrap .ln {
    width: 50px;
    min-width: 50px;
    max-width: 50px;
    text-align: right;
    padding: 0 8px;
    color: var(--text-faint);
    background: var(--bg-code);
    border-right: 1px solid var(--border-input);
    user-select: none;
    font-size: 11px;
}

.diff-table-wrap .sg {
    width: 20px;
    min-width: 20px;
    max-width: 20px;
    text-align: center;
    padding: 0;
    user-select: none;
    font-weight: 700;
}

.diff-table-wrap .ln-eq { background: var(--bg-code); }
.diff-table-wrap .eq { background: var(--bg-elevated); }

.diff-table-wrap .ln-del { background: var(--red-bg-light); color: var(--red-text-dark); }
.diff-table-wrap .sg-del { background: var(--red-bg-light); color: var(--red-dark); }
.diff-table-wrap .del { background: var(--red-bg); color: var(--text-diff); }
.diff-table-wrap .del-word { background: var(--red-word); border-radius: 2px; }

.diff-table-wrap .ln-add { background: var(--green-light); color: var(--green-text-dark); }
.diff-table-wrap .sg-add { background: var(--green-light); color: var(--green-dark); }
.diff-table-wrap .add { background: var(--green-bg-light); color: var(--text-diff); }
.diff-table-wrap .add-word { background: var(--green-word); border-radius: 2px; }

.diff-table-wrap .ln-ws { background: var(--purple-bg-light); color: var(--purple-text); }
.diff-table-wrap .sg-ws { background: var(--purple-bg-light); color: var(--purple-text-light); }
.diff-table-wrap .ws { background: var(--purple-bg); color: var(--text-diff); }
.diff-table-wrap .ws .del-word { background: var(--purple-word); border-radius: 2px; }
.diff-table-wrap .ws .add-word { background: var(--purple-word); border-radius: 2px; }

.diff-table-wrap tr.row-hidden { display: none; }

.diff-table-wrap tr.row-hl td { background: var(--yellow-bg) !important; }
.diff-table-wrap td.ln:not(:empty) { cursor: pointer; }
.diff-table-wrap td.ln:not(:empty):hover { text-decoration: underline; }

/* Inline expand button rows (GitHub-style full-row) */
.diff-table-wrap .btn-row td {
    text-align: center;
    padding: 4px 8px;
    background: var(--blue-bg-light, #ddf4ff);
    color: var(--accent-blue);
    cursor: pointer;
    font-size: 12px;
    border-top: 1px solid var(--accent-blue);
    border-bottom: 1px solid var(--accent-blue);
    user-select: none;
    line-height: 20px;
}
.diff-table-wrap .btn-row td:hover { background: var(--blue-bg-hover, #c8e9ff); }
.diff-table-wrap .btn-row.all-expanded td { display: none; }
.diff-table-wrap .btn-row .exp-arrow { font-weight: 700; font-size: 14px; padding: 0 2px; }
.diff-table-wrap .btn-row .exp-label { padding: 0 4px; }

/* ---- Manual alignment mode (Beyond Compare style) ---- */
.align-status {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 999;
    padding: 8px 20px;
    font-size: 13px;
    font-family: 'Segoe UI', system-ui, sans-serif;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.align-status.active { display: block; }
.align-status.mode-left { background: var(--align-pending-bg); color: var(--align-pending-text); }
.align-status.mode-right { background: var(--align-right-bg); color: var(--align-right-text); }
.align-status kbd {
    display: inline-block;
    padding: 1px 6px;
    font-size: 11px;
    font-family: inherit;
    background: rgba(0,0,0,0.08);
    border: 1px solid rgba(0,0,0,0.15);
    border-radius: 3px;
    margin: 0 2px;
}

.diff-table-wrap td.align-pending {
    outline: 2px solid var(--amber);
    outline-offset: -2px;
    background: var(--align-pending-bg) !important;
}
.diff-table-wrap td.align-pending-sibling {
    background: var(--align-pending-bg) !important;
}

.diff-table-wrap td.align-locked {
    outline: 2px solid var(--accent-blue);
    outline-offset: -2px;
    background: var(--align-locked-bg) !important;
}
.diff-table-wrap td.align-locked-sibling {
    background: var(--align-locked-bg) !important;
}

.diff-table-wrap.waiting-right td[data-side="r"]:not(:empty) {
    cursor: crosshair;
}
.diff-table-wrap.waiting-right td[data-side="r"]:not(:empty):hover {
    outline: 2px dashed var(--accent-blue);
    outline-offset: -2px;
}

.diff-table-wrap tr.row-aligned {
    border-left: 3px solid var(--amber);
}

.diff-toolbar {
    display: flex;
    gap: 6px;
    margin-bottom: 6px;
    padding: 6px 8px;
    background: var(--bg-code);
    border: 1px solid var(--border-input);
    border-bottom: none;
    border-radius: 6px 6px 0 0;
}
.diff-toolbar + .diff-table-wrap { border-radius: 0 0 6px 6px; }
.diff-toolbar button {
    padding: 3px 10px;
    font-size: 12px;
    cursor: pointer;
    background: var(--bg-elevated);
    border: 1px solid var(--border-input);
    border-radius: 4px;
    color: var(--text-diff);
    line-height: 20px;
    transition: background 0.1s;
}
.diff-toolbar button:hover:not(:disabled) { background: var(--bg-btn-hover); }
.diff-toolbar button:disabled { opacity: 0.35; cursor: default; }

.btn-copy { float: right; }
.copy-spacer { flex: 1; }
.copy-toast {
    position: fixed;
    bottom: 24px;
    right: 24px;
    background: var(--toast-bg);
    color: var(--bg);
    padding: 8px 18px;
    border-radius: 6px;
    font-size: 13px;
    z-index: 9999;
    animation: toastFade 1.5s ease forwards;
    pointer-events: none;
}
@keyframes toastFade {
    0%,60% { opacity: 1; }
    100% { opacity: 0; }
}
"""

_JS = """
var _activeFilter = null;

function toggleTheme() {
    var html = document.documentElement;
    var btn = document.getElementById('theme-btn');
    if (!btn) return;
    if (html.getAttribute('data-theme') === 'dark') {
        html.removeAttribute('data-theme');
        btn.textContent = '\\u263E Dark';
        localStorage.setItem('lower-trace-theme', 'light');
    } else {
        html.setAttribute('data-theme', 'dark');
        btn.textContent = '\\u2600 Light';
        localStorage.setItem('lower-trace-theme', 'dark');
    }
}

function filterByBadge(badgeEl) {
    var filter = badgeEl.getAttribute('data-filter');
    var bar = badgeEl.closest('.summary-bar');
    var allBadges = bar.querySelectorAll('.badge');

    if (_activeFilter === filter || filter === 'all') {
        _activeFilter = null;
    } else {
        _activeFilter = filter;
    }

    allBadges.forEach(function(b) {
        b.classList.remove('active', 'dimmed');
        if (_activeFilter) {
            if (b.getAttribute('data-filter') === _activeFilter) {
                b.classList.add('active');
            } else if (b.getAttribute('data-filter') !== 'all') {
                b.classList.add('dimmed');
            }
        }
    });

    var sidebar = document.querySelector('.sidebar:not([style*="display: none"])');
    if (!sidebar) sidebar = document.querySelector('.sidebar');
    if (!sidebar) return;

    var links = sidebar.querySelectorAll('.pass-link');
    var firstVisible = null;

    links.forEach(function(link) {
        var status = link.getAttribute('data-status');
        if (!_activeFilter || status === _activeFilter) {
            link.classList.remove('filtered-out');
            if (!firstVisible) firstVisible = link;
        } else {
            link.classList.add('filtered-out');
        }
    });

    if (firstVisible) {
        var activeLink = sidebar.querySelector('.pass-link.active');
        if (!activeLink || activeLink.classList.contains('filtered-out')) {
            var sid = firstVisible.getAttribute('data-target');
            showPass(firstVisible, sid);
        }
    }
}

function showPass(el, id) {
    document.querySelectorAll('.pass-section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.pass-link').forEach(l => l.classList.remove('active'));
    var sec = document.getElementById(id);
    if (sec) sec.classList.add('active');
    if (el) el.classList.add('active');
    if (typeof cancelAlign === 'function') cancelAlign();
    if (sec) {
        var tbl = sec.querySelector('table');
        if (tbl && typeof updBtns === 'function') updBtns(tbl);
    }
}

function showPhase(el, phase) {
    document.querySelectorAll('.phase-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.sidebar').forEach(s => s.style.display = 'none');
    document.querySelectorAll('.summary-bar').forEach(s => s.style.display = 'none');
    if (el) el.classList.add('active');
    var sb = document.getElementById('sb-' + phase);
    if (sb) sb.style.display = '';
    var sm = document.getElementById('sm-' + phase);
    if (sm) sm.style.display = '';
    document.querySelectorAll('.pass-section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.pass-link').forEach(l => l.classList.remove('active'));
    var first = document.querySelector('[data-phase="' + phase + '"]');
    if (first) {
        var sid = first.getAttribute('data-target');
        showPass(first, sid);
    }
}

function toggleIr(btn) {
    var section = btn.closest('.pass-section');
    var block = section ? section.querySelector('.ir-block') : null;
    if (block) {
        block.classList.toggle('show');
        btn.textContent = block.classList.contains('show')
            ? '\\u25BC Collapse IR' : '\\u25B6 Show full IR';
    }
}

function toggleIrLine(ln) {
    var line = ln.closest('.ir-line');
    var block = ln.closest('.ir-block');
    if (!line || !block) return;
    var wasHl = line.classList.contains('hl');
    block.querySelectorAll('.ir-line.hl').forEach(function(l) { l.classList.remove('hl'); });
    if (!wasHl) line.classList.add('hl');
}

function toggleCollapse(el) {
    var sec = el.closest('.pass-section');
    sec.classList.toggle('collapsed');
}

function copyIr(btn, side) {
    var sec = btn.closest('.pass-section');
    var el = sec.querySelector('.ir-data-' + side);
    if (!el) return;
    var text = el.textContent;
    var label = side === 'before' ? 'Before' : side === 'after' ? 'After' : 'IR';
    function showToast() {
        var toast = document.createElement('div');
        toast.className = 'copy-toast';
        toast.textContent = 'Copied ' + label + ' IR';
        document.body.appendChild(toast);
        setTimeout(function() { toast.remove(); }, 1600);
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(showToast).catch(function() {
            fallbackCopy(text, showToast);
        });
    } else {
        fallbackCopy(text, showToast);
    }
}
function fallbackCopy(text, cb) {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.style.cssText = 'position:fixed;left:-9999px';
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); if (cb) cb(); } catch(e) {}
    ta.remove();
}

function expand(el, n, evt) {
    var run = el.dataset.run;
    var tr = el.closest('tr');
    var table = tr.closest('table');
    var limit = (evt && evt.altKey) ? 999999 : n;
    var shown = 0, r;
    r = tr.previousElementSibling;
    while (r && shown < limit) {
        if (!r.classList.contains('row-hidden') || r.dataset.run !== run) break;
        r.classList.remove('row-hidden'); shown++;
        r = r.previousElementSibling;
    }
    shown = 0;
    r = tr.nextElementSibling;
    while (r && shown < limit) {
        if (!r.classList.contains('row-hidden') || r.dataset.run !== run) break;
        r.classList.remove('row-hidden'); shown++;
        r = r.nextElementSibling;
    }
    updBtns(table);
}

function expandAll(btn) {
    var table = btn.closest('.pass-section').querySelector('table');
    table.querySelectorAll('tr.row-hidden').forEach(function(r) { r.classList.remove('row-hidden'); });
    updBtns(table);
}

function collapseCtx(btn) {
    var table = btn.closest('.pass-section').querySelector('table');
    table.querySelectorAll('tr[data-collapse="1"]').forEach(function(r) {
        if (!r.classList.contains('btn-row')) r.classList.add('row-hidden');
    });
    updBtns(table);
}

function updBtns(table) {
    var sec = table.closest('.pass-section');
    var hid = table.querySelectorAll('tr.row-hidden');
    var ba = sec.querySelector('.btn-expand-all');
    if (hid.length === 0) {
        table.querySelectorAll('.btn-row').forEach(function(r) { r.classList.add('all-expanded'); });
        if (ba) ba.disabled = true;
        return;
    }
    table.querySelectorAll('.btn-row td[data-run]').forEach(function(td) {
        var run = td.dataset.run;
        var tr = td.closest('tr');
        var hasHidden = false;
        var r = tr.previousElementSibling;
        while (r) {
            if (r.classList.contains('row-hidden') && r.dataset.run === run) { hasHidden = true; break; }
            if (!r.classList.contains('row-hidden') || r.dataset.run !== run) break;
            r = r.previousElementSibling;
        }
        if (!hasHidden) {
            r = tr.nextElementSibling;
            while (r) {
                if (r.classList.contains('row-hidden') && r.dataset.run === run) { hasHidden = true; break; }
                if (!r.classList.contains('row-hidden') || r.dataset.run !== run) break;
                r = r.nextElementSibling;
            }
        }
        tr.style.display = hasHidden ? '' : 'none';
    });
    if (ba) ba.disabled = false;
}

/* ---- F7 Manual alignment (Beyond Compare style) ---- */
var _alignMode = null;
var _pendingLeft = null;
var _alignStatus = null;

function cancelAlign() {
    if (_pendingLeft) {
        var row = _pendingLeft.closest('tr');
        if (row) row.querySelectorAll('.align-pending,.align-pending-sibling,.align-locked,.align-locked-sibling')
            .forEach(function(c){ c.classList.remove('align-pending','align-pending-sibling','align-locked','align-locked-sibling'); });
    }
    document.querySelectorAll('.align-locked,.align-locked-sibling').forEach(function(c){ c.classList.remove('align-locked','align-locked-sibling'); });
    document.querySelectorAll('.waiting-right').forEach(function(c){ c.classList.remove('waiting-right'); });
    _pendingLeft = null;
    _alignMode = null;
    if (_alignStatus) { _alignStatus.className = 'align-status'; _alignStatus.innerHTML = ''; }
}

function showAlignStatus(mode, msg) {
    if (!_alignStatus) return;
    _alignStatus.className = 'align-status active mode-' + mode;
    _alignStatus.innerHTML = msg;
}

function alignRows(leftTd, rightTd) {
    var lIdx = parseInt(leftTd.getAttribute('data-idx'), 10);
    var rIdx = parseInt(rightTd.getAttribute('data-idx'), 10);
    var section = leftTd.closest('.pass-section');
    var beforeLines = getBeforeLines(section);
    var afterLines = getAfterLines(section);
    if (isNaN(lIdx) || isNaN(rIdx)) return;

    var upper = lcsOpcodes(beforeLines.slice(0, lIdx), afterLines.slice(0, rIdx), 0, 0);
    var lower = lcsOpcodes(beforeLines.slice(lIdx + 1), afterLines.slice(rIdx + 1), lIdx + 1, rIdx + 1);
    var mid = [{ tag: 'replace', i1: lIdx, i2: lIdx + 1, j1: rIdx, j2: rIdx + 1 }];
    var opcodes = upper.concat(mid, lower);

    var wrap = section.querySelector('.diff-table-wrap');
    var table = wrap.querySelector('table');
    table.innerHTML = buildDiffRowsHtml(opcodes, beforeLines, afterLines, lIdx, rIdx);
}

function getBeforeLines(section) {
    var el = section.querySelector('.ir-data-before');
    return el ? el.textContent.split('\\n') : [];
}
function getAfterLines(section) {
    var el = section.querySelector('.ir-data-after');
    return el ? el.textContent.split('\\n') : [];
}

function lcsOpcodes(a, b, aOff, bOff) {
    var m = a.length, n = b.length;
    var dp = [];
    for (var i = 0; i <= m; i++) { dp[i] = new Array(n + 1).fill(0); }
    for (var i = m - 1; i >= 0; i--) {
        for (var j = n - 1; j >= 0; j--) {
            dp[i][j] = (a[i] === b[j]) ? dp[i+1][j+1] + 1 : Math.max(dp[i+1][j], dp[i][j+1]);
        }
    }
    var blocks = [], i = 0, j = 0;
    while (i < m && j < n) {
        if (a[i] === b[j]) {
            var si = i, sj = j;
            while (i < m && j < n && a[i] === b[j]) { i++; j++; }
            blocks.push([si, sj, i - si]);
        } else if (dp[i+1][j] >= dp[i][j+1]) { i++; } else { j++; }
    }
    var ops = [], ai = 0, bi = 0;
    for (var k = 0; k < blocks.length; k++) {
        var si = blocks[k][0], sj = blocks[k][1], size = blocks[k][2];
        if (si > ai || sj > bi) {
            var tag = (si > ai && sj > bi) ? 'replace' : (si > ai ? 'delete' : 'insert');
            ops.push({ tag: tag, i1: ai + aOff, i2: si + aOff, j1: bi + bOff, j2: sj + bOff });
        }
        ops.push({ tag: 'equal', i1: si + aOff, i2: si + size + aOff, j1: sj + bOff, j2: sj + size + bOff });
        ai = si + size; bi = sj + size;
    }
    if (ai < m || bi < n) {
        var tag = (ai < m && bi < n) ? 'replace' : (ai < m ? 'delete' : 'insert');
        ops.push({ tag: tag, i1: ai + aOff, i2: m + aOff, j1: bi + bOff, j2: n + bOff });
    }
    return ops;
}

function buildDiffRowsHtml(opcodes, beforeLines, afterLines, pinL, pinR) {
    var s = '<colgroup><col style="width:50px"><col style="width:20px"><col><col style="width:50px"><col style="width:20px"><col></colgroup>';
    var rows = [];
    for (var o = 0; o < opcodes.length; o++) {
        var op = opcodes[o];
        if (op.tag === 'equal') {
            for (var i = op.i1; i < op.i2; i++) {
                var j = op.j1 + (i - op.i1);
                rows.push('<tr>' +
                    '<td class="ln ln-eq" data-side="l" data-idx="' + i + '">' + (i+1) + '</td>' +
                    '<td class="sg"></td><td class="eq">' + escHtml(beforeLines[i]||'') + '</td>' +
                    '<td class="ln ln-eq" data-side="r" data-idx="' + j + '">' + (j+1) + '</td>' +
                    '<td class="sg"></td><td class="eq">' + escHtml(afterLines[j]||'') + '</td></tr>');
            }
        } else if (op.tag === 'replace') {
            // Monotone matching: LCS on stripped content guarantees both
            // line-number columns render in ascending order (no sort needed).
            var bStripped = [], aStripped = [];
            for (var x = op.i1; x < op.i2; x++) bStripped.push((beforeLines[x]||'').trim());
            for (var y = op.j1; y < op.j2; y++) aStripped.push((afterLines[y]||'').trim());
            var subOps = lcsOpcodes(bStripped, aStripped, op.i1, op.j1);
            var matched = [];
            for (var k = 0; k < subOps.length; k++) {
                var so = subOps[k];
                if (so.tag === 'equal') {
                    for (var t = 0; t < so.i2 - so.i1; t++) matched.push([so.i1 + t, so.j1 + t]);
                }
            }
            var mL = {}, mR = {};
            for (var k = 0; k < matched.length; k++) { mL[matched[k][0]] = true; mR[matched[k][1]] = true; }
            var unmatchedL = [], unmatchedR = [];
            for (var k = op.i1; k < op.i2; k++) { if (!mL[k]) unmatchedL.push(k); }
            for (var k = op.j1; k < op.j2; k++) { if (!mR[k]) unmatchedR.push(k); }
            var all = [];
            var up = 0, vp = 0;
            var flushGap = function(gl, gr) {
                for (var k = 0; k < Math.max(gl.length, gr.length); k++) {
                    if (k < gl.length && k < gr.length) all.push([gl[k], gr[k], false]);
                    else if (k < gl.length) all.push([gl[k], null, false]);
                    else all.push([null, gr[k], false]);
                }
            };
            for (var mi = 0; mi < matched.length; mi++) {
                var li = matched[mi][0], ri = matched[mi][1];
                var gl = [], gr = [];
                while (up < unmatchedL.length && unmatchedL[up] < li) { gl.push(unmatchedL[up]); up++; }
                while (vp < unmatchedR.length && unmatchedR[vp] < ri) { gr.push(unmatchedR[vp]); vp++; }
                flushGap(gl, gr);
                all.push([li, ri, true]);
            }
            flushGap(unmatchedL.slice(up), unmatchedR.slice(vp));
            for (var k = 0; k < all.length; k++) {
                var li = all[k][0], ri = all[k][1], matched = all[k][2];
                var isPinned = (li === pinL && ri === pinR);
                if (li !== null && ri !== null) {
                    var diff = jsInlineDiff(beforeLines[li]||'', afterLines[ri]||'');
                    if (diff.isWsOnly && matched) {
                        var cls = isPinned ? ' class="row-aligned"' : '';
                        rows.push('<tr' + cls + '>' +
                            '<td class="ln ln-ws" data-side="l" data-idx="' + li + '">' + (li+1) + '</td>' +
                            '<td class="sg sg-ws">~</td><td class="ws">' + diff.left + '</td>' +
                            '<td class="ln ln-ws" data-side="r" data-idx="' + ri + '">' + (ri+1) + '</td>' +
                            '<td class="sg sg-ws">~</td><td class="ws">' + diff.right + '</td></tr>');
                    } else {
                        var cls = isPinned ? ' class="row-aligned"' : '';
                        rows.push('<tr' + cls + '>' +
                            '<td class="ln ln-del" data-side="l" data-idx="' + li + '">' + (li+1) + '</td>' +
                            '<td class="sg sg-del">\u2212</td><td class="del">' + diff.left + '</td>' +
                            '<td class="ln ln-add" data-side="r" data-idx="' + ri + '">' + (ri+1) + '</td>' +
                            '<td class="sg sg-add">+</td><td class="add">' + diff.right + '</td></tr>');
                    }
                } else if (li !== null) {
                    rows.push('<tr>' +
                        '<td class="ln ln-del" data-side="l" data-idx="' + li + '">' + (li+1) + '</td>' +
                        '<td class="sg sg-del">\u2212</td><td class="del">' + escHtml(beforeLines[li]||'') + '</td>' +
                        '<td class="ln"></td><td class="sg"></td><td></td></tr>');
                } else {
                    rows.push('<tr>' +
                        '<td class="ln"></td><td class="sg"></td><td></td>' +
                        '<td class="ln ln-add" data-side="r" data-idx="' + ri + '">' + (ri+1) + '</td>' +
                        '<td class="sg sg-add">+</td><td class="add">' + escHtml(afterLines[ri]||'') + '</td></tr>');
                }
            }
        } else if (op.tag === 'delete') {
            for (var i = op.i1; i < op.i2; i++) {
                rows.push('<tr>' +
                    '<td class="ln ln-del" data-side="l" data-idx="' + i + '">' + (i+1) + '</td>' +
                    '<td class="sg sg-del">\u2212</td><td class="del">' + escHtml(beforeLines[i]||'') + '</td>' +
                    '<td class="ln"></td><td class="sg"></td><td></td></tr>');
            }
        } else if (op.tag === 'insert') {
            for (var j = op.j1; j < op.j2; j++) {
                rows.push('<tr>' +
                    '<td class="ln"></td><td class="sg"></td><td></td>' +
                    '<td class="ln ln-add" data-side="r" data-idx="' + j + '">' + (j+1) + '</td>' +
                    '<td class="sg sg-add">+</td><td class="add">' + escHtml(afterLines[j]||'') + '</td></tr>');
            }
        }
    }
    return s + rows.join('');
}

function jsInlineDiff(before, after) {
    var m = before.length, n = after.length;
    var left = [], right = [], isWsOnly = true;
    var sm = [];
    for (var i = 0; i <= m; i++) {
        sm[i] = [];
        for (var j = 0; j <= n; j++) sm[i][j] = 0;
    }
    for (var i = m - 1; i >= 0; i--) {
        for (var j = n - 1; j >= 0; j--) {
            if (before[i] === after[j]) sm[i][j] = sm[i+1][j+1] + 1;
            else sm[i][j] = Math.max(sm[i+1][j], sm[i][j+1]);
        }
    }
    var i = 0, j = 0;
    while (i < m && j < n) {
        if (before[i] === after[j]) {
            left.push(escHtml(before[i]));
            right.push(escHtml(after[j]));
            i++; j++;
        } else if (sm[i+1] && sm[i+1][j] >= sm[i][j+1]) {
            var chunk = before[i];
            i++;
            while (i < m && (j >= n || (sm[i+1] && sm[i+1][j] >= sm[i][j+1])) && before[i] !== after[j]) {
                chunk += before[i]; i++;
            }
            if (chunk.trim() !== '') isWsOnly = false;
            left.push('<span class="del-word">' + escHtml(chunk) + '</span>');
        } else {
            var chunk = after[j];
            j++;
            while (j < n && (i >= m || sm[i][j+1] > sm[i+1][j]) && before[i] !== after[j]) {
                chunk += after[j]; j++;
            }
            if (chunk.trim() !== '') isWsOnly = false;
            right.push('<span class="add-word">' + escHtml(chunk) + '</span>');
        }
    }
    if (i < m) {
        var rest = before.slice(i);
        if (rest.trim() !== '') isWsOnly = false;
        left.push('<span class="del-word">' + escHtml(rest) + '</span>');
    }
    if (j < n) {
        var rest = after.slice(j);
        if (rest.trim() !== '') isWsOnly = false;
        right.push('<span class="add-word">' + escHtml(rest) + '</span>');
    }
    return { left: left.join(''), right: right.join(''), isWsOnly: isWsOnly };
}

function escHtml(text) {
    return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function initSidebarResize() {
    var handle = document.getElementById('sidebar-resize');
    var openBtn = document.getElementById('sidebar-open-btn');
    if (!handle) return;

    var sidebar = document.querySelector('.sidebar:not([style*="display: none"])') || document.querySelector('.sidebar');
    if (!sidebar) return;

    var dragging = false, startX = 0, startW = 0;

    handle.addEventListener('mousedown', function(e) {
        if (sidebar.classList.contains('collapsed')) return;
        dragging = true;
        startX = e.clientX;
        startW = sidebar.offsetWidth;
        handle.classList.add('active');
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
        e.preventDefault();
    });

    document.addEventListener('mousemove', function(e) {
        if (!dragging) return;
        var w = Math.max(150, Math.min(600, startW + e.clientX - startX));
        sidebar.style.width = w + 'px';
    });

    document.addEventListener('mouseup', function() {
        if (!dragging) return;
        dragging = false;
        handle.classList.remove('active');
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
    });
}

function toggleSidebar(btn) {
    var sidebars = document.querySelectorAll('.sidebar');
    var handle = document.getElementById('sidebar-resize');
    var openBtn = document.getElementById('sidebar-open-btn');
    sidebars.forEach(function(s) { s.classList.add('collapsed'); });
    if (handle) handle.style.display = 'none';
    if (openBtn) openBtn.style.display = '';
    document.querySelectorAll('.sidebar-toggle-btn.inside').forEach(function(b) { b.style.display = 'none'; });
}

function openSidebar() {
    var sidebars = document.querySelectorAll('.sidebar');
    var handle = document.getElementById('sidebar-resize');
    var openBtn = document.getElementById('sidebar-open-btn');
    sidebars.forEach(function(s) { s.classList.remove('collapsed'); s.style.width = ''; });
    if (handle) handle.style.display = '';
    if (openBtn) openBtn.style.display = 'none';
    document.querySelectorAll('.sidebar-toggle-btn.inside').forEach(function(b) { b.style.display = ''; });
}
"""


def render_pass_section(rec: LowerRecord) -> str:
    """Render a single pass section as HTML."""
    sid = f"sec-{rec.phase}-{rec.index}"

    if rec.status == STATUS_FAILED:
        error_html = f'<div class="error-box"><div class="error-label">Exception</div>{_esc(rec.error_msg)}</div>'
        if rec.before_text:
            _ir_lines = rec.before_text.splitlines()
            _ln_width = len(str(len(_ir_lines)))
            _ir_with_ln = "".join(
                f'<span class="ir-line"><span class="ir-ln" onclick="toggleIrLine(this)">{str(i + 1).rjust(_ln_width)}</span>{_esc(line)}</span>'
                for i, line in enumerate(_ir_lines)
            )
            before_content = (
                '<p class="noop-msg">IR before this pass (execution failed):</p>'
                '<button class="ir-toggle" onclick="toggleIr(this)">&#9654; Show before IR</button>'
                f'<div class="ir-block"><pre>{_ir_with_ln}</pre></div>'
            )
        else:
            before_content = ""
        status_html = '<span class="status status-failed">✘ FAILED</span>'
        return (
            f'<div class="pass-section failed-section" id="{sid}">'
            f'<div class="pass-header">'
            f"<h2>{rec.index:02d}. {rec.name}</h2>"
            f"{status_html}"
            f"</div>"
            f"{error_html}"
            f"{before_content}"
            f'<pre hidden class="ir-data-before">{_esc(rec.before_text) if rec.before_text else ""}</pre>'
            f"</div>"
        )

    elif rec.status == STATUS_SKIPPED:
        status_html = '<span class="status status-skipped">— SKIPPED</span>'
        return (
            f'<div class="pass-section skipped-section" id="{sid}">'
            f'<div class="pass-header">'
            f"<h2>{rec.index:02d}. {rec.name}</h2>"
            f"{status_html}"
            f"</div>"
            f'<p class="noop-msg">This pass did not run (a previous pass failed).</p>'
            f"</div>"
        )

    elif rec.changed:
        diff_html = _make_diff_html(rec.before_text, rec.after_text, context=3)
        if rec.status == STATUS_CODEGEN:
            lang_labels = (
                '<div class="codegen-lang-bar">'
                '<span class="codegen-lang-label tir">Lowered TIR</span>'
                '<span class="codegen-lang-label cpp">Generated C++</span>'
                "</div>"
            )
            diff_content = (
                f'<div class="diff-toolbar">'
                f'<button class="btn-expand-all" '
                f'onclick="expandAll(this)">⊞ Show all context</button>'
                f'<button onclick="collapseCtx(this)">⊟ Collapse</button>'
                f'<span class="copy-spacer"></span>'
                f'<button class="btn-copy" onclick="copyIr(this,\'before\')">📋 Copy TIR</button>'
                f'<button class="btn-copy" onclick="copyIr(this,\'after\')">📋 Copy C++</button>'
                f"</div>"
                f"{lang_labels}"
                f"{diff_html}"
            )
            status_html = '<span class="status status-codegen">CODEGEN</span>'
            return (
                f'<div class="pass-section codegen-section" id="{sid}">'
                f'<div class="pass-header collapsible" onclick="toggleCollapse(this)">'
                f'<span class="pass-toggle"></span>'
                f"<h2>{rec.index:02d}. {rec.name}</h2>"
                f"{status_html}"
                f"</div>"
                f"{diff_content}"
                f'<pre hidden class="ir-data-before">{_esc(rec.before_text)}</pre>\n'
                f'<pre hidden class="ir-data-after">{_esc(rec.after_text)}</pre>\n'
                f"</div>"
            )
        diff_content = (
            f'<div class="diff-toolbar">'
            f'<button class="btn-expand-all" '
            f'onclick="expandAll(this)">⊞ Show all context</button>'
            f'<button onclick="collapseCtx(this)">⊟ Collapse</button>'
            f'<span class="copy-spacer"></span>'
            f'<button class="btn-copy" onclick="copyIr(this,\'before\')">📋 Copy Before</button>'
            f'<button class="btn-copy" onclick="copyIr(this,\'after\')">📋 Copy After</button>'
            f"</div>"
            f"{diff_html}"
        )
        status_html = '<span class="status status-changed">CHANGED</span>'
        return (
            f'<div class="pass-section" id="{sid}">'
            f'<div class="pass-header collapsible" onclick="toggleCollapse(this)">'
            f'<span class="pass-toggle"></span>'
            f"<h2>{rec.index:02d}. {rec.name}</h2>"
            f"{status_html}"
            f"</div>"
            f"{diff_content}"
            f'<pre hidden class="ir-data-before">{_esc(rec.before_text)}</pre>'
            f'<pre hidden class="ir-data-after">{_esc(rec.after_text)}</pre>'
            f"</div>"
        )
    else:
        _ir_lines = rec.after_text.splitlines()
        _ln_width = len(str(len(_ir_lines)))
        _ir_with_ln = "".join(
            f'<span class="ir-line"><span class="ir-ln" onclick="toggleIrLine(this)">{str(i + 1).rjust(_ln_width)}</span>{_esc(line)}</span>'
            for i, line in enumerate(_ir_lines)
        )
        diff_content = (
            '<p class="noop-msg">This pass did not modify the IR.</p>'
            '<button class="ir-toggle" onclick="toggleIr(this)">&#9654; Show full IR</button>'
            '<button class="btn-copy" onclick="copyIr(this,\'after\')">📋 Copy IR</button>'
            f'<div class="ir-block"><pre>{_ir_with_ln}</pre></div>'
        )
        status_html = '<span class="status status-noop">NO-OP</span>'
        return (
            f'<div class="pass-section" id="{sid}">'
            f'<div class="pass-header">'
            f"<h2>{rec.index:02d}. {rec.name}</h2>"
            f"{status_html}"
            f"</div>"
            f"{diff_content}"
            f'<pre hidden class="ir-data-after">{_esc(rec.after_text)}</pre>'
            f"</div>"
        )


def generate_html(records: list[LowerRecord], output_path: str):
    """Generate a self-contained HTML file with pass trace visualization."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    phases: dict = {}
    for rec in records:
        phases.setdefault(rec.phase, []).append(rec)

    phase_tabs_html = []
    sidebars_html = []
    summaries_html = []
    sections_html = []

    for pi, (phase_name, phase_records) in enumerate(phases.items()):
        is_active = pi == 0
        active_cls = " active" if is_active else ""
        active_style = "" if is_active else ' style="display:none"'

        pretty_phase = (
            phase_name.replace("_", " ")
            .replace("phase1", "Phase 1:")
            .replace("phase2", "Phase 2:")
            .replace("pipeline", "Pipeline:")
            .replace("codegen", "Codegen")
        )
        pretty_phase = pretty_phase.strip()
        if pretty_phase and pretty_phase[0].islower():
            pretty_phase = pretty_phase[0].upper() + pretty_phase[1:]

        n_total = len(phase_records)
        n_completed = sum(1 for r in phase_records if r.status == STATUS_COMPLETED)
        n_changed = sum(1 for r in phase_records if r.changed and r.status != STATUS_CODEGEN)
        n_codegen = sum(1 for r in phase_records if r.status == STATUS_CODEGEN)
        n_failed = sum(1 for r in phase_records if r.status == STATUS_FAILED)
        n_skipped = sum(1 for r in phase_records if r.status == STATUS_SKIPPED)
        n_noop = n_completed - n_changed

        phase_tabs_html.append(f'<div class="phase-tab{active_cls}" onclick="showPhase(this, \'{phase_name}\')">{pretty_phase}</div>')

        failed_badge = ""
        skipped_badge = ""
        codegen_badge = ""
        if n_failed:
            failed_badge = f'<span class="badge badge-failed" data-filter="failed" onclick="filterByBadge(this)">✘ {n_failed} failed</span>'
        if n_skipped:
            skipped_badge = (
                f'<span class="badge badge-skipped" data-filter="skipped" onclick="filterByBadge(this)">— {n_skipped} skipped</span>'
            )
        if n_codegen and n_codegen < n_total:
            codegen_badge = (
                f'<span class="badge badge-codegen" data-filter="codegen" onclick="filterByBadge(this)">{n_codegen} codegen</span>'
            )
        summaries_html.append(
            f'<div class="summary-bar" id="sm-{phase_name}"{active_style}>'
            f'<span class="badge badge-total" data-filter="all" onclick="filterByBadge(this)">{n_total} passes</span>'
            f'<span class="badge badge-changed" data-filter="changed" onclick="filterByBadge(this)">{n_changed} changed</span>'
            f'<span class="badge badge-noop" data-filter="noop" onclick="filterByBadge(this)">{n_noop} no-op</span>'
            f"{codegen_badge}"
            f"{failed_badge}"
            f"{skipped_badge}"
            f"</div>"
        )

        links = []
        for rec in phase_records:
            if rec.status == STATUS_FAILED:
                dot_cls = "failed"
                status_attr = "failed"
            elif rec.status == STATUS_SKIPPED:
                dot_cls = "skipped"
                status_attr = "skipped"
            elif rec.status == STATUS_CODEGEN:
                dot_cls = "codegen"
                status_attr = "codegen"
            elif rec.changed:
                dot_cls = "changed"
                status_attr = "changed"
            else:
                dot_cls = "noop"
                status_attr = "noop"
            sid = f"sec-{rec.phase}-{rec.index}"
            stats_html = ""
            if rec.changed:
                stats_html = (
                    f'<span class="pass-stats">'
                    f'<span class="st-add">+{rec.add_lines}</span> '
                    f'<span class="st-del">−{rec.del_lines}</span>'
                    f"</span>"
                )
            elif rec.status == STATUS_FAILED:
                stats_html = '<span class="pass-stats"><span class="st-del">ERROR</span></span>'
            elif rec.status == STATUS_SKIPPED:
                stats_html = '<span class="pass-stats" style="color:#94a3b8">—</span>'
            links.append(
                f'<a class="pass-link" data-phase="{rec.phase}" data-target="{sid}" '
                f'data-status="{status_attr}" '
                f"onclick=\"showPass(this, '{sid}')\">"
                f'<span class="pass-idx">{rec.index:02d}</span>'
                f'<span class="pass-dot {dot_cls}"></span>'
                f'<span class="pass-label">{rec.name}</span>'
                f"{stats_html}"
                f"</a>"
            )
        sidebars_html.append(
            f'<div class="sidebar" id="sb-{phase_name}"{active_style}>'
            f'<button class="sidebar-toggle-btn inside" onclick="toggleSidebar(this)" title="Collapse sidebar"><span class="chevron"></span></button>'
            f'<div class="section-title">Passes</div>' + "\n".join(links) + "</div>"
        )

        for rec in phase_records:
            sections_html.append(render_pass_section(rec))

    auto_show_js = ""
    if records:
        first_sid = f"sec-{records[0].phase}-{records[0].index}"
        auto_show_js = (
            "document.addEventListener('DOMContentLoaded', function() {\n"
            f"  var firstSid = '{first_sid}';\n"
            "  var el = document.querySelector('[data-target=\"' + firstSid + '\"]');\n"
            "  if (el) showPass(el, firstSid);\n"
            "  if (typeof initSidebarResize === 'function') initSidebarResize();\n"
            "  _alignStatus = document.getElementById('align-status');\n"
            "  var _saved = localStorage.getItem('lower-trace-theme');\n"
            "  if (_saved === 'dark') {\n"
            "    document.documentElement.setAttribute('data-theme', 'dark');\n"
            "    var _tb = document.getElementById('theme-btn');\n"
            "    if (_tb) _tb.textContent = '\\u2600 Light';\n"
            "  }\n"
            "\n"
            "  var passLinks = document.getElementsByClassName('pass-link');\n"
            "  var activeLink = el || null;\n"
            "\n"
            "  document.addEventListener('keydown', function(e) {\n"
            "    var tag = (e.target.tagName || '').toLowerCase();\n"
            "    if (tag === 'input' || tag === 'textarea' || tag === 'select') return;\n"
            "\n"
            "    if (e.key === 'j' || e.key === 'k') {\n"
            "      e.preventDefault();\n"
            "      var idx = -1;\n"
            "      for (var i = 0; i < passLinks.length; i++) {\n"
            "        if (passLinks[i] === activeLink) { idx = i; break; }\n"
            "      }\n"
            "      if (e.key === 'j' && idx < passLinks.length - 1) idx++;\n"
            "      else if (e.key === 'k' && idx > 0) idx--;\n"
            "      else return;\n"
            "      var next = passLinks[idx];\n"
            "      var phase = next.getAttribute('data-phase');\n"
            "      var curTab = document.querySelector('.phase-tab.active');\n"
            "      if (!curTab || curTab.textContent.indexOf(phase.replace('phase1','Phase 1').replace('phase2','Phase 2')) < 0) {\n"
            "        var tabs = document.querySelectorAll('.phase-tab');\n"
            "        for (var t = 0; t < tabs.length; t++) {\n"
            "          if (tabs[t].getAttribute('onclick').indexOf(phase) > -1) {\n"
            "            showPhase(tabs[t], phase); break;\n"
            "          }\n"
            "        }\n"
            "      }\n"
            "      activeLink = next;\n"
            "      var sid = next.getAttribute('data-target');\n"
            "      showPass(next, sid);\n"
            "      next.scrollIntoView({block: 'nearest'});\n"
            "      var sec = document.getElementById(sid);\n"
            "      if (sec) sec.scrollIntoView({block: 'start'});\n"
            "    }\n"
            "\n"
            "    if (e.key === 'E' && e.shiftKey && !e.ctrlKey && !e.metaKey) {\n"
            "      e.preventDefault();\n"
            "      document.querySelectorAll('.pass-section table tr.row-hidden').forEach(function(r) {\n"
            "        r.classList.remove('row-hidden');\n"
            "      });\n"
            "      document.querySelectorAll('.pass-section table').forEach(function(tbl) { updBtns(tbl); });\n"
            "    }\n"
            "\n"
            "    if (e.key === 'F7') {\n"
            "      e.preventDefault();\n"
            "      if (_alignMode === null) {\n"
            "        _alignMode = 'left';\n"
            "        showAlignStatus('left', 'Align: Click a <b>left</b> (before) line number &nbsp; <kbd>F7</kbd> to confirm &nbsp; <kbd>Esc</kbd> to cancel');\n"
            "      } else if (_alignMode === 'left' && _pendingLeft) {\n"
            "        _alignMode = 'right';\n"
            "        _pendingLeft.classList.remove('align-pending');\n"
            "        _pendingLeft.classList.add('align-locked');\n"
            "        var sib = _pendingLeft.nextElementSibling;\n"
            "        while (sib && sib.getAttribute('data-side') !== 'r') {\n"
            "          if (sib.getAttribute('data-side') === 'l') sib.classList.add('align-locked-sibling');\n"
            "          sib = sib.nextElementSibling;\n"
            "        }\n"
            "        var wrap = _pendingLeft.closest('.diff-table-wrap');\n"
            "        if (wrap) wrap.classList.add('waiting-right');\n"
            "        showAlignStatus('right', 'Align: Click a <b>right</b> (after) line number to align &nbsp; <kbd>Esc</kbd> to cancel');\n"
            "      } else if (_alignMode === 'right') {\n"
            "        cancelAlign();\n"
            "      }\n"
            "      return;\n"
            "    }\n"
            "\n"
            "    if (e.key === 'Escape' && _alignMode) {\n"
            "      e.preventDefault();\n"
            "      cancelAlign();\n"
            "      return;\n"
            "    }\n"
            "  });\n"
            "\n"
            "  document.addEventListener('click', function(e) {\n"
            "    if (!_alignMode) return;\n"
            "    var td = e.target.closest('td[data-side]');\n"
            "    if (!td) return;\n"
            "    var section = td.closest('.pass-section.active');\n"
            "    if (!section) return;\n"
            "    if (_alignMode === 'left' && td.getAttribute('data-side') === 'l' && td.textContent.trim() !== '') {\n"
            "      e.stopPropagation();\n"
            "      if (_pendingLeft) {\n"
            "        var oldRow = _pendingLeft.closest('tr');\n"
            "        if (oldRow) oldRow.querySelectorAll('.align-pending,.align-pending-sibling')\n"
            "          .forEach(function(c){ c.classList.remove('align-pending','align-pending-sibling'); });\n"
            "      }\n"
            "      _pendingLeft = td;\n"
            "      td.classList.add('align-pending');\n"
            "      var sib = td.nextElementSibling;\n"
            "      while (sib && sib.getAttribute('data-side') !== 'r') {\n"
            "        if (sib.getAttribute('data-side') === 'l') sib.classList.add('align-pending-sibling');\n"
            "        sib = sib.nextElementSibling;\n"
            "      }\n"
            "      showAlignStatus('left', 'Left line <b>' + td.textContent.trim() + '</b> selected &nbsp; Press <kbd>F7</kbd> to confirm &nbsp; <kbd>Esc</kbd> to cancel');\n"
            "      return;\n"
            "    }\n"
            "    if (_alignMode === 'right' && td.getAttribute('data-side') === 'r' && td.textContent.trim() !== '' && _pendingLeft) {\n"
            "      e.stopPropagation();\n"
            "      alignRows(_pendingLeft, td);\n"
            "      cancelAlign();\n"
            "      return;\n"
            "    }\n"
            "  }, true);\n"
            "\n"
            "  document.addEventListener('click', function(e) {\n"
            "    var td = e.target.closest('td[data-side]');\n"
            "    if (!td) return;\n"
            "    var tr = td.closest('tr');\n"
            "    if (!tr || tr.closest('.pass-section.active') === null) return;\n"
            "    var wasHl = tr.classList.contains('row-hl');\n"
            "    document.querySelectorAll('tr.row-hl').forEach(function(r) { r.classList.remove('row-hl'); });\n"
            "    if (!wasHl) tr.classList.add('row-hl');\n"
            "  });\n"
            "});\n"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TileLang Lower Trace</title>
<style>{_CSS}</style>
</head>
<body>
<div class="align-status" id="align-status"></div>
<div class="header">
  <h1>TileLang Lower Trace</h1>
  <div class="sub">Compilation pipeline visualization &middot; {len(records)} passes recorded</div>
  <button class="theme-btn" id="theme-btn" onclick="toggleTheme()" title="Toggle dark/light theme">&#9788; Dark</button>
</div>
<div class="phase-tabs">
  {"".join(phase_tabs_html)}
</div>
{"".join(summaries_html)}
<div class="main">
  {"".join(sidebars_html)}
  <div class="sidebar-resize" id="sidebar-resize"></div>
  <button class="sidebar-toggle-btn" id="sidebar-open-btn" onclick="openSidebar()" style="display:none" title="Show sidebar"><span class="chevron"></span></button>
  <div class="content">
    {"".join(sections_html)}
  </div>
</div>
<script>
{_JS}
{auto_show_js}
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
