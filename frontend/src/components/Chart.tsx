/* One small chart, drawn as SVG in code. A single amber series on a muted
   grid — the chart is a moment, not a dashboard. Supports the types the
   backend recommender returns: bar, line, scatter, histogram, bar_counts. */

import type { ChartSpec, Row } from "../api";

interface Props {
  chart: ChartSpec;
  rows: Row[];
}

const W = 720;
const H = 260;
const PAD_L = 56;
const PAD_R = 16;
const PAD_T = 16;
const PAD_B = 36;

function fmt(n: number): string {
  if (Math.abs(n) >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (Math.abs(n) >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return Number.isInteger(n) ? String(n) : n.toFixed(1);
}

function truncate(s: string, n = 16): string {
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

export default function Chart({ chart, rows }: Props) {
  const { type, x, y } = chart;
  if (!x || rows.length === 0) return null;

  const innerW = W - PAD_L - PAD_R;
  const innerH = H - PAD_T - PAD_B;

  if (type === "bar" || type === "bar_counts") {
    const data = rows.slice(0, 12).map((r) => ({
      label: String(r[x] ?? ""),
      value:
        type === "bar_counts"
          ? rows.filter((o) => String(o[x]) === String(r[x])).length
          : Number(r[y ?? ""] ?? 0),
    }));
    const uniq = type === "bar_counts"
      ? [...new Map(data.map((d) => [d.label, d])).values()].slice(0, 12)
      : data;
    const max = Math.max(...uniq.map((d) => d.value), 1);
    const slot = innerW / uniq.length;
    const bw = Math.min(44, slot * 0.6);
    return (
      <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg" role="img">
        {[0, 0.5, 1].map((f) => (
          <g key={f}>
            <line
              x1={PAD_L} x2={W - PAD_R}
              y1={PAD_T + innerH * (1 - f)} y2={PAD_T + innerH * (1 - f)}
              className="chart-grid"
            />
            <text x={PAD_L - 8} y={PAD_T + innerH * (1 - f) + 4}
              textAnchor="end" className="chart-tick">{fmt(max * f)}</text>
          </g>
        ))}
        {uniq.map((d, i) => {
          const h = (d.value / max) * innerH;
          const cx = PAD_L + slot * i + slot / 2;
          return (
            <g key={i}>
              <rect x={cx - bw / 2} y={PAD_T + innerH - h} width={bw}
                height={h} rx={3} className="chart-bar" />
              <text x={cx} y={H - 12} textAnchor="middle" className="chart-tick">
                {truncate(d.label)}
              </text>
            </g>
          );
        })}
      </svg>
    );
  }

  if (type === "line" && y) {
    const pts = rows.slice(0, 40).map((r) => Number(r[y] ?? 0));
    const max = Math.max(...pts, 1);
    const min = Math.min(...pts, 0);
    const span = max - min || 1;
    const step = innerW / Math.max(pts.length - 1, 1);
    const coords = pts.map(
      (v, i) =>
        `${PAD_L + step * i},${PAD_T + innerH - ((v - min) / span) * innerH}`,
    );
    return (
      <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg" role="img">
        {[0, 0.5, 1].map((f) => (
          <g key={f}>
            <line x1={PAD_L} x2={W - PAD_R}
              y1={PAD_T + innerH * (1 - f)} y2={PAD_T + innerH * (1 - f)}
              className="chart-grid" />
            <text x={PAD_L - 8} y={PAD_T + innerH * (1 - f) + 4}
              textAnchor="end" className="chart-tick">{fmt(min + span * f)}</text>
          </g>
        ))}
        <polyline points={coords.join(" ")} className="chart-line" />
        {coords.map((p, i) => {
          const [cx, cy] = p.split(",").map(Number);
          return <circle key={i} cx={cx} cy={cy} r={3} className="chart-dot" />;
        })}
      </svg>
    );
  }

  if (type === "scatter" && y) {
    const xs = rows.map((r) => Number(r[x] ?? 0));
    const ys = rows.map((r) => Number(r[y] ?? 0));
    const xMax = Math.max(...xs, 1);
    const xMin = Math.min(...xs, 0);
    const yMax = Math.max(...ys, 1);
    const yMin = Math.min(...ys, 0);
    return (
      <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg" role="img">
        {[0, 0.5, 1].map((f) => (
          <g key={f}>
            <line x1={PAD_L} x2={W - PAD_R}
              y1={PAD_T + innerH * (1 - f)} y2={PAD_T + innerH * (1 - f)}
              className="chart-grid" />
            <text x={PAD_L - 8} y={PAD_T + innerH * (1 - f) + 4}
              textAnchor="end" className="chart-tick">{fmt(yMin + (yMax - yMin) * f)}</text>
          </g>
        ))}
        {rows.slice(0, 200).map((r, i) => (
          <circle
            key={i}
            cx={PAD_L + ((Number(r[x] ?? 0) - xMin) / (xMax - xMin || 1)) * innerW}
            cy={PAD_T + innerH - ((Number(r[y] ?? 0) - yMin) / (yMax - yMin || 1)) * innerH}
            r={4} className="chart-dot"
          />
        ))}
      </svg>
    );
  }

  if (type === "histogram") {
    const vals = rows.map((r) => Number(r[x] ?? 0)).filter((v) => !Number.isNaN(v));
    if (vals.length === 0) return null;
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const bins = 10;
    const counts = new Array(bins).fill(0);
    vals.forEach((v) => {
      const b = Math.min(bins - 1, Math.floor(((v - min) / (max - min || 1)) * bins));
      counts[b] += 1;
    });
    const cMax = Math.max(...counts, 1);
    const bw = innerW / bins;
    return (
      <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg" role="img">
        {counts.map((c: number, i: number) => (
          <g key={i}>
            <rect x={PAD_L + bw * i + 1} y={PAD_T + innerH - (c / cMax) * innerH}
              width={bw - 2} height={(c / cMax) * innerH} rx={2} className="chart-bar" />
          </g>
        ))}
        <text x={PAD_L} y={H - 12} className="chart-tick">{fmt(min)}</text>
        <text x={W - PAD_R} y={H - 12} textAnchor="end" className="chart-tick">{fmt(max)}</text>
      </svg>
    );
  }

  return null;
}
