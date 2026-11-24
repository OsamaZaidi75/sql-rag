"""Print the rolling-N-day golden-eval accuracy trend.

Reads the per-run history that ``evals/run_evals.py`` appends to
``reports/daily/eval_history.csv`` (columns: date,passed,total,accuracy_pct,llm)
and prints a per-day summary for the most recent ``--days`` days, so a bad
run is visible as a dip in the trend rather than a single red number.

Run:  python scripts/eval_trend.py [--history PATH] [--days N]
      [--stale-days N] [--min-runs N] [--min-avg PCT] [--min-worst PCT]
      [--min-latest PCT] [--min-best PCT] [--min-spread PCT] [--min-days N] [--min-llms N]
      [--min-provider-runs N]
      [--by-llm] [--quiet] [--json PATH]

JSON output: pass --json PATH to also write the window summary, the
per-day summaries, and the per-provider breakdown as a JSON artifact
(parent dirs auto-created), so CI can archive the trend instead of
parsing the prose lines — same family as the ``--json`` verdict in
``evals/run_evals.py``. The breakdown is always included in the JSON,
whether or not ``--by-llm`` was passed for the prose output; blank
llm cells group under "-" like the per-day lines. The window summary
also names the worst provider (``worst_provider`` plus
``worst_provider_accuracy``, picked from the same worst-first order
as the breakdown), so a CI dashboard gets the riskiest provider
without parsing the prose summary line. It also names the best
provider (``best_provider`` plus ``best_provider_accuracy`` — the
last provider in that same worst-first order, i.e. the one whose
worst run was the highest), so both ends of the provider spread are
visible without parsing the prose. It also records the spread
itself (``spread`` — the best provider's worst run minus the worst
provider's worst run, the same value the ``--min-spread`` gate
checks), so a CI dashboard gets the provider divergence without
parsing prose. It also names the thinnest provider
(``thinnest_provider`` plus ``thinnest_provider_runs`` — the
provider with the fewest runs in the window, ties naming the
alphabetically first provider, the same value the
``--min-provider-runs`` gate checks), so a CI dashboard sees the
thinnest provider without parsing prose. It also names the latest
provider (``latest_provider`` plus ``latest_provider_accuracy`` —
the provider behind the window's newest run, the same value the
summary already prints as ``latest`` and the ``--min-latest`` gate
checks), so a CI dashboard sees the provider behind the latest run
without parsing prose. Each ``by_llm`` entry also records the provider's
``share`` of the window's runs (percent, 1 decimal — the same value
the prose breakdown prints as ``{n} of {total} runs``), so a CI
dashboard gets the weighting without parsing prose. The file is written
even when the window holds no runs (``runs`` 0, empty ``days`` and
``by_llm``, ``worst_provider``, ``best_provider``,
``thinnest_provider`` and ``latest_provider`` null), and a
write failure exits 2.

Archiving the trend: pass --json reports/trend.json to write the JSON
artifact under the repo, e.g.
``python scripts/eval_trend.py --json reports/trend.json`` — wiring the
same invocation into ``.github/workflows/daily-maintenance.yml`` is NOT
possible via the API (the GitHub App lacks the workflows permission),
so a human running the trend manually archives the JSON instead.

Staleness check (recommended for a manual run): pass --stale-days 7
to flag a silently-dead scheduled workflow — e.g.
``python scripts/eval_trend.py --stale-days 7`` warns on stderr and
exits 1 when the newest recorded run is more than 7 days old, while
still printing the trend. A fresh history (newest row within 7 days)
exits 0.

Minimum-runs check: pass --min-runs N to warn on stderr and exit 1
when the window holds fewer than N runs, while still printing the
trend — so a thin history can't masquerade as a healthy trend.

Minimum-average check: pass --min-avg PCT to warn on stderr and exit 1
when the window's average accuracy drops below PCT, while still
printing the trend — so a trend that is fresh and thick but bad still
shows red. An empty window fails the check (no average to meet the
floor).

Minimum-worst check: pass --min-worst PCT to warn on stderr and exit 1
when the window's worst single run drops below PCT, while still
printing the trend — so one bad run can't hide inside a healthy
average. An empty window fails the check (no worst run to meet the
floor).

Minimum-latest check: pass --min-latest PCT to warn on stderr and exit 1
when the window's latest run drops below PCT, while still printing
the trend — so a regressing latest run shows red even when older runs
were perfect. An empty window fails the check (no latest run to meet
the floor).

Minimum-best check: pass --min-best PCT to warn on stderr and exit 1
when the best provider's worst run drops below PCT, while still
printing the trend — so a window where even the best provider has a
bad run shows red. The best provider is the last provider in the same
worst-first order as the --by-llm breakdown (highest worst run, ties
naming the alphabetically last provider) — the same value the --json
window summary reports as ``best_provider_accuracy``. An empty window
fails the check (no best provider to meet the floor).

Minimum-spread check: pass --min-spread PCT to warn on stderr and
exit 1 when the spread between the best and worst provider's worst
run exceeds PCT, while still printing the trend — so a window where
providers diverge sharply shows red even when every absolute gate
passes. The spread is the best provider's worst run minus the worst
provider's worst run (the same values the --json window summary
reports as ``best_provider_accuracy`` and ``worst_provider_accuracy``,
and the summary records the spread itself as ``spread``);
a single-provider window has spread 0.0. An empty window fails the
check (no providers to compare).

Minimum-days check: pass --min-days N to warn on stderr and exit 1
when the window spans fewer than N distinct days, while still
printing the trend — so clustered same-day runs can't masquerade as
a multi-day trend. An empty window fails the check (no days covered).

Minimum-llms check: pass --min-llms N to warn on stderr and exit 1
when the window's runs come from fewer than N distinct LLM values,
while still printing the trend — so a single-provider streak can't
masquerade as a diverse trend. An empty window fails the check
(no LLMs seen). Blank llm cells don't count as a provider.

Minimum-provider-runs check: pass --min-provider-runs N to warn on
stderr and exit 1 when any provider in the window has fewer than N
runs, while still printing the trend — so a provider with a single
lucky run can't hide inside a thick window (the window can pass
--min-runs on total runs while one provider barely ran at all). An
empty window fails the check (no providers to count). Blank llm cells
group under "-" like the per-day lines, so an unnamed provider counts
as one provider.

Per-LLM breakdown: pass --by-llm to print, under the window summary,
one line per provider (its share of the window's runs — e.g.
`gemini: 1 of 3 runs` — average accuracy, worst run, and latest
run), so a mixed-provider window's makeup is visible instead of just
its distinct-LLM count. Providers are listed worst-first (lowest worst
accuracy first, ties broken alphabetically), so the riskiest provider
surfaces first. The window summary line also names that worst provider
(e.g. `worst provider: zeta (80.0%)`), so it stays visible even in
`--quiet` mode without reading the breakdown lines. It names the best
provider too (e.g. `best provider: alpha (100.0%)`) — the last provider
in the same worst-first order, i.e. the one whose worst run was the
highest, ties naming the alphabetically last provider — so both ends
of the provider spread stay visible in `--quiet` mode. It also names
the spread itself (e.g. `spread: 20.0%` — the best provider's worst
run minus the worst provider's worst run, the same value the JSON
window summary reports as ``spread`` and the ``--min-spread`` gate
checks), so the provider divergence is visible on the summary line
without reading the breakdown lines. It also names the thinnest
provider (e.g. `thinnest provider: gemini (1 run)` — the provider
with the fewest runs in the window, ties naming the alphabetically
first provider, the same value the JSON window summary reports as
``thinnest_provider`` / ``thinnest_provider_runs`` and the
``--min-provider-runs`` gate checks), so the thinnest provider stays
visible on the summary line without reading the breakdown lines.
It also names the latest provider (e.g. `latest provider: gemini
(100.0%)` — the provider behind the window's newest run, the same
value the summary already prints as `latest` and the `--min-latest`
gate checks), so the provider behind the latest run stays visible on
the summary line without reading the breakdown lines.
Blank llm cells group under "-" like the per-day lines.

Quiet mode: pass --quiet to print only the window summary line (and
the per-provider breakdown when --by-llm is also passed), skipping
the per-day lines — so a healthy trend stays one line in CI logs
instead of one line per day. Gates still run and still exit 1 on a
miss; only the prose is shortened.

Exit codes: 0 on success, 1 when --stale-days is given and the newest
recorded run is older than that many days (or there are no runs at all),
when --min-runs is given and the window holds fewer than N runs, when
--min-avg is given and the window average is below PCT (or there are
no runs at all), or when --min-worst is given and the window's worst
run is below PCT (or there are no runs at all), when --min-latest is
given and the window's latest run is below PCT (or there are no runs
at all), when --min-best is given and the best provider's worst run
is below PCT (or there are no runs at all), when --min-spread is
given and the spread between the best and worst provider's worst
run exceeds PCT (or there are no runs at all), when --min-days is given
and the window spans fewer than N distinct days (or there are no runs
at all), when --min-llms is
given and the window's runs come from fewer than N distinct LLM
values (or there are no runs at all), when --min-provider-runs is
given and any provider in the window has fewer than N runs (or there
are no runs at all), 2 when the history file
is missing or unreadable.
"""

import argparse
import csv
import json
import os
import sys
from datetime import date, timedelta

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_HISTORY = os.path.join(REPO_ROOT, "reports", "daily", "eval_history.csv")
DEFAULT_DAYS = 7


def parse_date(value: str) -> date:
    """Parse a YYYY-MM-DD date cell; raises ValueError on malformed input."""
    year, month, day = (int(part) for part in value.strip().split("-"))
    return date(year, month, day)


def read_history(path: str):
    """Read eval history rows as dicts; skips malformed rows with a warning.

    Returns a list of {"date": date, "passed": int, "total": int,
    "accuracy": float, "llm": str} sorted by date ascending.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {"date", "passed", "total", "accuracy_pct"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"history file {path!r} is missing columns: {sorted(missing)}")
        rows = []
        for lineno, row in enumerate(reader, start=2):
            try:
                rows.append(
                    {
                        "date": parse_date(row["date"]),
                        "passed": int(row["passed"]),
                        "total": int(row["total"]),
                        "accuracy": float(row["accuracy_pct"]),
                        "llm": (row.get("llm") or "").strip(),
                    }
                )
            except (ValueError, TypeError, AttributeError) as exc:
                print(f"warning: skipping malformed row {lineno}: {exc}", file=sys.stderr)
    rows.sort(key=lambda r: r["date"])
    return rows


def rolling_window(rows, days: int):
    """Return the rows inside the window [latest - (days-1), latest].

    The window is anchored on the newest date in the history rather than on
    today's date, so the trend still works when the workflow run is delayed.
    """
    if not rows:
        return []
    latest = rows[-1]["date"]
    cutoff = latest - timedelta(days=days - 1)
    return [r for r in rows if r["date"] >= cutoff]


def summarize(rows):
    """Summarize one day's rows: run count, avg/min/latest accuracy."""
    if not rows:
        return {"runs": 0, "avg_accuracy": 0.0, "min_accuracy": 0.0, "latest_accuracy": 0.0}
    accuracies = [r["accuracy"] for r in rows]
    return {
        "runs": len(rows),
        "avg_accuracy": round(sum(accuracies) / len(accuracies), 1),
        "min_accuracy": min(accuracies),
        "latest_accuracy": accuracies[-1],
    }


def is_stale(latest, today: date, stale_days: int) -> bool:
    """True when the newest recorded run is older than ``stale_days``.

    ``latest`` is the newest date in the history (or None when there are no
    runs — an empty history is stale by definition). A future-dated latest
    row is never stale.
    """
    if latest is None:
        return True
    return (today - latest).days > stale_days


def _providers_by_worst(by_provider):
    """Order providers worst-first for the per-LLM breakdown.

    ``by_provider`` maps provider -> list of accuracies. The provider
    with the lowest worst run comes first, so the riskiest provider
    surfaces first in a mixed-provider window; ties break
    alphabetically so the order is deterministic.
    """
    return sorted(by_provider, key=lambda p: (min(by_provider[p]), p))


def render(day_rows, window_days, by_llm=False, quiet=False):
    """Render the trend as plain text lines (one per day + a summary).

    The summary line also shows the distinct-LLM count, so the
    ``--min-llms`` gate's input is visible at a glance (blank llm
    cells don't count, same as the per-day ``llm=`` lists). With
    ``by_llm`` true, the summary line also names the worst provider
    (lowest worst run, ties broken alphabetically — the same
    worst-first order as the breakdown) and the best provider
    (highest worst run, ties broken alphabetically last — the last
    provider in that same order), plus the spread between them
    (best provider's worst run minus worst provider's worst run,
    the same value the JSON window summary reports as ``spread``)
    and the thinnest provider (fewest runs in the window, ties
    broken alphabetically first — the same value the JSON window
    summary reports as ``thinnest_provider`` /
    ``thinnest_provider_runs``), so both ends of the provider
    spread, the divergence between them, and the thinnest provider
    stay visible even when ``quiet`` hides the per-day lines and
    only the summary prints. It also names the latest provider
    (the provider behind the window's newest run — the same value
    the summary already prints as ``latest`` and the
    ``--min-latest`` gate checks), so the provider behind the
    latest run stays visible on the summary line too. A per-provider breakdown follows the
    summary (blank llm cells group under "-", like the per-day lines),
    with each provider's share of the window's runs, average accuracy,
    worst run, and latest run, so a provider with a high average but
    one bad run stays
    visible, and its current state is visible without re-running the
    window. The breakdown is ordered worst-first (see
    ``_providers_by_worst``). With ``quiet`` true, the per-day lines
    are skipped — only the window summary line (and the breakdown when
    ``by_llm`` is also true) is printed, so a healthy trend stays one
    line in CI logs instead of one line per day.
    """
    lines = []
    total_runs = 0
    all_accuracies = []
    window_llms = set()
    for day, rows in day_rows:
        total_runs += len(rows)
        all_accuracies.extend(r["accuracy"] for r in rows)
        window_llms.update(r["llm"] for r in rows if r["llm"])
        if quiet:
            continue
        summary = summarize(rows)
        llms = ",".join(sorted({r["llm"] for r in rows if r["llm"]}) or ["-"])
        lines.append(
            f"{day.isoformat()}  runs={summary['runs']:>2}  "
            f"avg={summary['avg_accuracy']:>5.1f}%  min={summary['min_accuracy']:>5.1f}%  "
            f"llm={llms}"
        )
    if all_accuracies:
        window_avg = round(sum(all_accuracies) / len(all_accuracies), 1)
        summary_line = (
            f"-- {window_days}-day window: {total_runs} runs, "
            f"avg {window_avg:.1f}%, worst {min(all_accuracies):.1f}%, "
            f"latest {all_accuracies[-1]:.1f}%, llms {len(window_llms)}"
        )
        by_provider = None
        if by_llm:
            by_provider = {}
            for day, rows in day_rows:
                for r in rows:
                    by_provider.setdefault(r["llm"] or "-", []).append(r["accuracy"])
            worst_provider = _providers_by_worst(by_provider)[0]
            summary_line += (
                f", worst provider: {worst_provider} "
                f"({min(by_provider[worst_provider]):.1f}%)"
            )
            best_provider = _providers_by_worst(by_provider)[-1]
            summary_line += (
                f", best provider: {best_provider} "
                f"({min(by_provider[best_provider]):.1f}%)"
            )
            spread = round(
                min(by_provider[best_provider]) - min(by_provider[worst_provider]),
                1,
            )
            summary_line += f", spread: {spread:.1f}%"
            # the thinnest provider has the fewest runs in the window —
            # the same value the JSON window summary reports as
            # ``thinnest_provider`` / ``thinnest_provider_runs`` and
            # the --min-provider-runs gate checks; ties name the
            # alphabetically first provider, mirroring the worst
            # provider's tie-break from the same (fewest runs, name)
            # ordering
            thinnest_provider = min(
                by_provider, key=lambda p: (len(by_provider[p]), p)
            )
            thinnest_runs = len(by_provider[thinnest_provider])
            summary_line += (
                f", thinnest provider: {thinnest_provider} "
                f"({thinnest_runs} run{'s' if thinnest_runs != 1 else ''})"
            )
            # the latest provider ran the window's newest run — the
            # same value the summary already prints as ``latest``
            # and the --min-latest gate checks; its provider is the
            # llm of the chronologically last row in the window
            # (blank cells group under "-"), so the provider behind
            # the latest run stays visible even in --quiet mode
            # without reading the breakdown lines
            latest_provider = day_rows[-1][1][-1]["llm"] or "-"
            summary_line += (
                f", latest provider: {latest_provider} "
                f"({all_accuracies[-1]:.1f}%)"
            )
        lines.append(summary_line)
        if by_llm:
            lines.append("by LLM:")
            for provider in _providers_by_worst(by_provider):
                accuracies = by_provider[provider]
                provider_avg = round(sum(accuracies) / len(accuracies), 1)
                lines.append(
                    f"  {provider}: {len(accuracies)} of {total_runs} runs, "
                    f"avg {provider_avg:.1f}%, worst {min(accuracies):.1f}%, "
                    f"latest {accuracies[-1]:.1f}%"
                )
    return "\n".join(lines)


def trend_dict(day_rows, window_days):
    """Build the machine-readable trend for ``--json``.

    Returns {"window": {...}, "days": [...], "by_llm": {...}}: the window
    summary (runs, average/worst/latest accuracy, distinct-LLM count,
    and the worst provider's name plus its worst accuracy, picked
    from the same worst-first order as the breakdown — plus the best
    provider's name plus its worst accuracy, the last provider in
    that same order, i.e. the one whose worst run was the highest —
    and the provider spread (``spread``, the best provider's worst
    run minus the worst provider's worst run, the same value the
    ``--min-spread`` gate checks) — and the thinnest provider's name
    plus its run count (``thinnest_provider`` /
    ``thinnest_provider_runs``, the fewest runs in the window, ties
    naming the alphabetically first provider — the same value the
    ``--min-provider-runs`` gate checks) — and the latest provider's
    name plus its accuracy (``latest_provider`` /
    ``latest_provider_accuracy``, the provider behind the window's
    newest run — the same value the summary prints as ``latest``
    and the ``--min-latest`` gate checks),
    one per-day summary (runs, avg/min/latest accuracy, llms seen),
    and one entry per provider (runs, share of the window's runs,
    average, worst, latest accuracy).
    The numbers mirror the prose ``render()`` output exactly, including
    the worst-first provider order (see ``_providers_by_worst``); the
    per-provider breakdown is always present here, whether or not
    ``--by-llm`` was passed for the prose output. Blank llm cells
    group under "-" like the per-day lines.
    """
    days = []
    all_accuracies = []
    window_llms = set()
    by_provider = {}
    for day, rows in day_rows:
        summary = summarize(rows)
        all_accuracies.extend(r["accuracy"] for r in rows)
        window_llms.update(r["llm"] for r in rows if r["llm"])
        days.append(
            {
                "date": day.isoformat(),
                "runs": summary["runs"],
                "avg_accuracy": summary["avg_accuracy"],
                "min_accuracy": summary["min_accuracy"],
                "latest_accuracy": summary["latest_accuracy"],
                "llms": sorted({r["llm"] for r in rows if r["llm"]}),
            }
        )
        for r in rows:
            by_provider.setdefault(r["llm"] or "-", []).append(r["accuracy"])
    providers = {}
    for provider in _providers_by_worst(by_provider):
        accuracies = by_provider[provider]
        providers[provider] = {
            "runs": len(accuracies),
            "share": round(100.0 * len(accuracies) / len(all_accuracies), 1),
            "avg_accuracy": round(sum(accuracies) / len(accuracies), 1),
            "worst_accuracy": min(accuracies),
            "latest_accuracy": accuracies[-1],
        }
    worst_provider = _providers_by_worst(by_provider)[0] if by_provider else None
    # the best provider is the last provider in the same worst-first
    # order — the one whose worst run was the highest — so both ends
    # of the provider spread come from one shared ordering; ties on
    # the highest worst run name the alphabetically last provider
    best_provider = _providers_by_worst(by_provider)[-1] if by_provider else None
    # the thinnest provider has the fewest runs in the window — the
    # same value the --min-provider-runs gate checks; ties name the
    # alphabetically first provider, mirroring the worst provider's
    # tie-break from the same (fewest runs, name) ordering
    thinnest_provider = (
        min(by_provider, key=lambda p: (len(by_provider[p]), p))
        if by_provider
        else None
    )
    # the latest provider ran the window's newest run — the same value
    # the summary already prints as ``latest`` and the --min-latest
    # gate checks; its provider is the llm of the chronologically
    # last row in the window (blank cells group under "-"), so the
    # provider behind the latest run is named even when it is not
    # the worst, best, or thinnest provider
    latest_provider = None
    if day_rows:
        latest_provider = day_rows[-1][1][-1]["llm"] or "-"
    return {
        "window": {
            "window_days": window_days,
            "runs": len(all_accuracies),
            "avg_accuracy": round(sum(all_accuracies) / len(all_accuracies), 1)
            if all_accuracies
            else 0.0,
            "worst_accuracy": min(all_accuracies) if all_accuracies else 0.0,
            "latest_accuracy": all_accuracies[-1] if all_accuracies else 0.0,
            "llms": len(window_llms),
            "worst_provider": worst_provider,
            "worst_provider_accuracy": (
                min(by_provider[worst_provider]) if worst_provider else 0.0
            ),
            "best_provider": best_provider,
            "best_provider_accuracy": (
                min(by_provider[best_provider]) if best_provider else 0.0
            ),
            "spread": (
                round(
                    min(by_provider[best_provider])
                    - min(by_provider[worst_provider]),
                    1,
                )
                if worst_provider and best_provider
                else 0.0
            ),
            "thinnest_provider": thinnest_provider,
            "thinnest_provider_runs": (
                len(by_provider[thinnest_provider]) if thinnest_provider else 0
            ),
            "latest_provider": latest_provider,
            "latest_provider_accuracy": (
                all_accuracies[-1] if all_accuracies else 0.0
            ),
        },
        "days": days,
        "by_llm": providers,
    }


def write_json(path: str, payload: dict) -> None:
    """Write a JSON artifact (stdlib only), creating parent dirs as needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def main(argv=None, today=None) -> int:
    """Run the trend report. ``today`` is injectable for tests; the CLI
    always uses the real current date as the staleness reference."""
    parser = argparse.ArgumentParser(
        description="Print the rolling-N-day golden-eval accuracy trend."
    )
    parser.add_argument(
        "--history",
        default=DEFAULT_HISTORY,
        help=f"path to eval_history.csv (default: {DEFAULT_HISTORY})",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help=f"rolling window in days (default: {DEFAULT_DAYS})",
    )
    parser.add_argument(
        "--stale-days",
        type=int,
        default=None,
        metavar="N",
        help="warn and exit 1 when the newest recorded run is older than N "
        "days (a silently-dead scheduled workflow shows up as a red flag "
        "instead of a quietly-shortening trend; unset disables the check)",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=None,
        metavar="N",
        help="warn and exit 1 when the rolling window holds fewer than N "
        "runs (a thin history can't masquerade as a healthy trend; "
        "unset disables the check)",
    )
    parser.add_argument(
        "--min-avg",
        type=float,
        default=None,
        metavar="PCT",
        help="warn and exit 1 when the rolling window's average accuracy "
        "is below PCT (a fresh, thick but bad trend still shows red; "
        "unset disables the check)",
    )
    parser.add_argument(
        "--min-worst",
        type=float,
        default=None,
        metavar="PCT",
        help="warn and exit 1 when the rolling window's worst single run "
        "is below PCT (one bad run can't hide inside a healthy average; "
        "unset disables the check)",
    )
    parser.add_argument(
        "--min-latest",
        type=float,
        default=None,
        metavar="PCT",
        help="warn and exit 1 when the rolling window's latest run "
        "is below PCT (a regressing latest run shows red even when "
        "older runs were perfect; unset disables the check)",
    )
    parser.add_argument(
        "--min-best",
        type=float,
        default=None,
        metavar="PCT",
        help="warn and exit 1 when the best provider's worst run "
        "is below PCT (a window where even the best provider has a "
        "bad run shows red; unset disables the check)",
    )
    parser.add_argument(
        "--min-spread",
        type=float,
        default=None,
        metavar="PCT",
        help="warn and exit 1 when the spread between the best and "
        "worst provider's worst run exceeds PCT (providers diverging "
        "sharply show red even when every absolute gate passes; "
        "unset disables the check)",
    )
    parser.add_argument(
        "--min-days",
        type=int,
        default=None,
        metavar="N",
        help="warn and exit 1 when the rolling window spans fewer "
        "than N distinct days (clustered same-day runs can't "
        "masquerade as a multi-day trend; unset disables the check)",
    )
    parser.add_argument(
        "--min-llms",
        type=int,
        default=None,
        metavar="N",
        help="warn and exit 1 when the rolling window's runs come "
        "from fewer than N distinct LLM values (a single-provider "
        "streak can't masquerade as a diverse trend; unset "
        "disables the check)",
    )
    parser.add_argument(
        "--min-provider-runs",
        type=int,
        default=None,
        metavar="N",
        help="warn and exit 1 when any provider in the rolling "
        "window has fewer than N runs (a provider with a single "
        "lucky run can't hide inside a thick window; unset "
        "disables the check)",
    )
    parser.add_argument(
        "--by-llm",
        action="store_true",
        help="print a per-provider breakdown (runs, average, worst "
        "and latest accuracy) under the window summary",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print only the window summary line (and the per-provider "
        "breakdown when --by-llm is also passed), skipping the per-day "
        "lines, so a healthy trend stays one line in CI logs",
    )
    parser.add_argument(
        "--json",
        default=None,
        metavar="PATH",
        help="write the window summary, per-day summaries and "
        "per-provider breakdown as a JSON artifact (parent dirs "
        "auto-created) so CI can archive the trend instead of "
        "parsing the prose lines",
    )
    args = parser.parse_args(argv)
    if args.days < 1:
        parser.error("--days must be a positive integer")
    if args.stale_days is not None and args.stale_days < 0:
        parser.error("--stale-days must be a non-negative integer")
    if args.min_runs is not None and args.min_runs < 0:
        parser.error("--min-runs must be a non-negative integer")
    if args.min_avg is not None and not 0 <= args.min_avg <= 100:
        parser.error("--min-avg must be between 0 and 100")
    if args.min_worst is not None and not 0 <= args.min_worst <= 100:
        parser.error("--min-worst must be between 0 and 100")
    if args.min_latest is not None and not 0 <= args.min_latest <= 100:
        parser.error("--min-latest must be between 0 and 100")
    if args.min_best is not None and not 0 <= args.min_best <= 100:
        parser.error("--min-best must be between 0 and 100")
    if args.min_spread is not None and not 0 <= args.min_spread <= 100:
        parser.error("--min-spread must be between 0 and 100")
    if args.min_days is not None and args.min_days < 0:
        parser.error("--min-days must be a non-negative integer")
    if args.min_llms is not None and args.min_llms < 0:
        parser.error("--min-llms must be a non-negative integer")
    if args.min_provider_runs is not None and args.min_provider_runs < 0:
        parser.error("--min-provider-runs must be a non-negative integer")
    try:
        rows = read_history(args.history)
    except (OSError, ValueError) as exc:
        print(f"error: cannot read history file: {exc}", file=sys.stderr)
        return 2
    exit_code = 0
    if args.stale_days is not None:
        reference = today if today is not None else date.today()
        latest = rows[-1]["date"] if rows else None
        if is_stale(latest, reference, args.stale_days):
            age = "no runs" if latest is None else f"{(reference - latest).days} days old"
            print(
                f"warning: eval history is stale: newest run is {age} "
                f"(threshold {args.stale_days} days)",
                file=sys.stderr,
            )
            exit_code = 1
    window = rolling_window(rows, args.days)
    if args.min_avg is not None:
        if not window:
            print(
                f"warning: eval history window holds no runs "
                f"(minimum average {args.min_avg:.1f}%)",
                file=sys.stderr,
            )
            exit_code = 1
        else:
            window_avg = round(
                sum(r["accuracy"] for r in window) / len(window), 1
            )
            if window_avg < args.min_avg:
                print(
                    f"warning: eval window average {window_avg:.1f}% is below "
                    f"minimum {args.min_avg:.1f}%",
                    file=sys.stderr,
                )
                exit_code = 1
    if args.min_worst is not None:
        if not window:
            print(
                f"warning: eval history window holds no runs "
                f"(minimum worst {args.min_worst:.1f}%)",
                file=sys.stderr,
            )
            exit_code = 1
        else:
            window_worst = min(r["accuracy"] for r in window)
            if window_worst < args.min_worst:
                print(
                    f"warning: eval window worst {window_worst:.1f}% is below "
                    f"minimum worst {args.min_worst:.1f}%",
                    file=sys.stderr,
                )
                exit_code = 1
    if args.min_latest is not None:
        if not window:
            print(
                f"warning: eval history window holds no runs "
                f"(minimum latest {args.min_latest:.1f}%)",
                file=sys.stderr,
            )
            exit_code = 1
        else:
            window_latest = window[-1]["accuracy"]
            if window_latest < args.min_latest:
                print(
                    f"warning: eval window latest {window_latest:.1f}% is below "
                    f"minimum latest {args.min_latest:.1f}%",
                    file=sys.stderr,
                )
                exit_code = 1
    if args.min_best is not None:
        if not window:
            print(
                f"warning: eval history window holds no runs "
                f"(minimum best {args.min_best:.1f}%)",
                file=sys.stderr,
            )
            exit_code = 1
        else:
            by_provider = {}
            for r in window:
                by_provider.setdefault(r["llm"] or "-", []).append(r["accuracy"])
            best_provider = _providers_by_worst(by_provider)[-1]
            best_worst = min(by_provider[best_provider])
            if best_worst < args.min_best:
                print(
                    f"warning: eval window best {best_worst:.1f}% is below "
                    f"minimum best {args.min_best:.1f}%",
                    file=sys.stderr,
                )
                exit_code = 1
    if args.min_spread is not None:
        if not window:
            print(
                f"warning: eval history window holds no runs "
                f"(maximum provider spread {args.min_spread:.1f}%)",
                file=sys.stderr,
            )
            exit_code = 1
        else:
            by_provider = {}
            for r in window:
                by_provider.setdefault(r["llm"] or "-", []).append(r["accuracy"])
            ordered = _providers_by_worst(by_provider)
            worst_worst = min(by_provider[ordered[0]])
            best_worst = min(by_provider[ordered[-1]])
            spread = round(best_worst - worst_worst, 1)
            if spread > args.min_spread:
                print(
                    f"warning: eval window provider spread {spread:.1f}% "
                    f"exceeds maximum {args.min_spread:.1f}%",
                    file=sys.stderr,
                )
                exit_code = 1
    if args.min_runs is not None and len(window) < args.min_runs:
        print(
            f"warning: eval history window holds only {len(window)} runs "
            f"(minimum {args.min_runs})",
            file=sys.stderr,
        )
        exit_code = 1
    if args.min_days is not None:
        window_days = len({r["date"] for r in window})
        if window_days < args.min_days:
            print(
                f"warning: eval history window spans only {window_days} "
                f"distinct day(s) (minimum {args.min_days})",
                file=sys.stderr,
            )
            exit_code = 1
    if args.min_llms is not None:
        window_llms = len({r["llm"] for r in window if r["llm"]})
        if window_llms < args.min_llms:
            print(
                f"warning: eval history window shows only {window_llms} "
                f"distinct LLM(s) (minimum {args.min_llms})",
                file=sys.stderr,
            )
            exit_code = 1
    if args.min_provider_runs is not None:
        if not window:
            print(
                f"warning: eval history window holds no runs "
                f"(minimum provider runs {args.min_provider_runs})",
                file=sys.stderr,
            )
            exit_code = 1
        else:
            by_provider = {}
            for r in window:
                by_provider.setdefault(r["llm"] or "-", []).append(r["accuracy"])
            thinnest = min(by_provider, key=lambda p: (len(by_provider[p]), p))
            thinnest_runs = len(by_provider[thinnest])
            if thinnest_runs < args.min_provider_runs:
                print(
                    f"warning: eval window provider {thinnest} has only "
                    f"{thinnest_runs} run(s) "
                    f"(minimum {args.min_provider_runs})",
                    file=sys.stderr,
                )
                exit_code = 1
    by_day = {}
    for row in window:
        by_day.setdefault(row["date"], []).append(row)
    days = [(day, by_day[day]) for day in sorted(by_day)]
    if args.json:
        try:
            write_json(args.json, trend_dict(days, args.days))
        except OSError as exc:
            print(f"error: cannot write trend JSON: {exc}", file=sys.stderr)
            return 2
        print(f"Trend JSON written to {args.json}")
    if not days:
        print("no eval runs in history")
        return exit_code
    print(render(days, args.days, by_llm=args.by_llm, quiet=args.quiet))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
