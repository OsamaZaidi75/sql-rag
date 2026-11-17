/* Minimal SQL highlighting: keywords in amber, strings in soft green,
   numbers in muted blue-grey. No dependency, no rainbow. */

const KEYWORDS = new Set([
  "SELECT", "FROM", "WHERE", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER",
  "FULL", "CROSS", "ON", "AND", "OR", "NOT", "IN", "IS", "NULL", "LIKE",
  "GROUP", "BY", "ORDER", "ASC", "DESC", "LIMIT", "OFFSET", "HAVING",
  "DISTINCT", "AS", "WITH", "CASE", "WHEN", "THEN", "ELSE", "END",
  "UNION", "ALL", "BETWEEN", "EXISTS", "COUNT", "SUM", "AVG", "MIN",
  "MAX", "ROUND", "COALESCE", "CAST", "STRFTIME", "DATE",
]);

const TOKEN_RE =
  /('(?:[^']|'')*')|(--[^\n]*)|\b(\d+(?:\.\d+)?)\b|\b([A-Za-z_][A-Za-z0-9_]*)\b|(\s+|.)/g;

export default function SqlCode({ sql }: { sql: string }) {
  const parts: React.ReactNode[] = [];
  let m: RegExpExecArray | null;
  let i = 0;
  TOKEN_RE.lastIndex = 0;
  while ((m = TOKEN_RE.exec(sql)) !== null) {
    const [tok, str, comment, num, word] = m;
    if (str) {
      parts.push(<span key={i} className="tok-str">{tok}</span>);
    } else if (comment) {
      parts.push(<span key={i} className="tok-com">{tok}</span>);
    } else if (num) {
      parts.push(<span key={i} className="tok-num">{tok}</span>);
    } else if (word && KEYWORDS.has(word.toUpperCase())) {
      parts.push(<span key={i} className="tok-kw">{tok}</span>);
    } else {
      parts.push(<span key={i}>{tok}</span>);
    }
    i += 1;
  }
  return (
    <pre className="sql-code">
      <code>{parts}</code>
    </pre>
  );
}
