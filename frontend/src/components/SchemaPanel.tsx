/* Schema explorer: tables as an accordion, columns with types and row
   counts, foreign keys listed under their table. */

import { useState } from "react";
import type { Schema } from "../api";

export default function SchemaPanel({ schema }: { schema: Schema | null }) {
  const [open, setOpen] = useState<string | null>(null);

  if (!schema) {
    return (
      <aside className="sidebar">
        <div className="side-section">
          <div className="side-label">Schema</div>
          <p className="muted">Loading schema…</p>
        </div>
      </aside>
    );
  }

  return (
    <aside className="sidebar">
      <div className="side-section">
        <div className="side-label">Database</div>
        <div className="db-name">{schema.database}</div>
        <div className="muted">
          {schema.table_count} tables · SQLite · read-only
        </div>
      </div>

      <div className="side-section">
        <div className="side-label">Schema</div>
        <ul className="table-list">
          {schema.tables.map((t) => {
            const isOpen = open === t.name;
            return (
              <li key={t.name} className="table-item">
                <button
                  className="table-head"
                  onClick={() => setOpen(isOpen ? null : t.name)}
                  aria-expanded={isOpen}
                >
                  <span className="table-caret">{isOpen ? "▾" : "▸"}</span>
                  <span className="table-name">{t.name}</span>
                  <span className="table-rows">{t.row_count} rows</span>
                </button>
                {isOpen && (
                  <div className="table-body">
                    <ul className="col-list">
                      {t.columns.map((c) => (
                        <li key={c.name} className="col-row">
                          <span className="col-name">
                            {c.name}
                            {c.is_pk ? <span className="pk-mark"> PK</span> : null}
                          </span>
                          <span className="col-type">{c.type}</span>
                        </li>
                      ))}
                    </ul>
                    {t.foreign_keys.length > 0 && (
                      <div className="fk-block">
                        <div className="fk-label">Foreign keys</div>
                        {t.foreign_keys.map((fk, i) => (
                          <div key={i} className="fk-row">
                            {fk.from_column} → {fk.to_table}.{fk.to_column}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      </div>

      <div className="side-section side-foot">
        <div className="side-label">How it works</div>
        <p className="muted">
          Hybrid retrieval picks the relevant tables, the schema is pruned to
          those tables plus one-hop foreign keys, and only that reaches the
          model. Every query is validated, read-only, and capped.
        </p>
      </div>
    </aside>
  );
}
