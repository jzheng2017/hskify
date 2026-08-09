/** Shared shadow-DOM styling for dictionary, selection, and speech UI. */
export const LOOKUP_CSS = `
.hskify-lookup {
  background: #fff;
  border: 1px solid #d1d5db;
  border-radius: 9px;
  box-shadow: 0 8px 28px rgb(0 0 0 / 24%);
  color: #111827;
  display: grid;
  font: 13px/1.4 system-ui, sans-serif;
  gap: 7px;
  max-height: calc(100vh - 16px);
  max-width: min(320px, calc(100% - 8px));
  min-width: 190px;
  overflow: auto;
  padding: 10px 12px;
  pointer-events: auto;
  position: absolute;
  text-align: left;
  user-select: text;
  z-index: 6;
}
.hskify-lookup[hidden] { display: none; }
.hskify-lookup-heading {
  align-items: center;
  display: flex;
  gap: 10px;
  justify-content: space-between;
}
.hskify-speak {
  appearance: none;
  background: #eff6ff;
  border: 1px solid #93c5fd;
  border-radius: 999px;
  color: #1d4ed8;
  cursor: pointer;
  flex: none;
  font: 600 11px/1 system-ui, sans-serif;
  padding: 6px 9px;
}
.hskify-speak[aria-pressed="true"] { background: #1d4ed8; color: #fff; }
.hskify-speak:focus-visible { outline: 2px solid #2563eb; outline-offset: 2px; }
.hskify-speak:disabled {
  background: #f3f4f6;
  border-color: #d1d5db;
  color: #6b7280;
  cursor: not-allowed;
}
.hskify-lookup-entry,
.hskify-lookup-context {
  border-top: 1px solid #e5e7eb;
  display: grid;
  gap: 2px;
  padding-top: 6px;
}
.hskify-lookup-entry span:last-child,
.hskify-lookup-context span:last-child { color: #4b5563; }
`
