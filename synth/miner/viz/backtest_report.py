"""
backtest_report.py — HTML report generator for backtest experiments.

Combines strategy comparison charts, score distributions, and
tabular summaries into a single HTML file.
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Optional


def generate_html_report(
    scan_results: list[dict],
    output_dir: str = "result/reports",
    title: str = "Backtest Report",
) -> str:
    """
    Generate an HTML report from backtest scan results.

    Args:
        scan_results: List of benchmark result dicts from BacktestRunner.scan_all().
        output_dir: Directory to save the report.
        title: Report title.

    Returns:
        Path to the generated HTML file.
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(output_dir, f"backtest_report_{ts}.html")

    # Build summary tables
    rows = []
    for r in sorted(scan_results, key=lambda x: (x.get("asset", ""), x.get("avg_score", float("inf")))):
        rows.append({
            "Strategy": r.get("strategy", "?"),
            "Asset": r.get("asset", "?"),
            "Frequency": r.get("frequency", "?"),
            "Avg Score": f"{r.get('avg_score', float('inf')):.4f}",
            "Median Score": f"{r.get('median_score', float('inf')):.4f}",
            "Min Score": f"{r.get('min_score', float('inf')):.4f}",
            "Max Score": f"{r.get('max_score', float('inf')):.4f}",
            "Success Rate": f"{r.get('successful_runs', 0)}/{r.get('num_runs', 0)}",
        })

    # Group by asset for rankings
    by_asset: dict[str, list[dict]] = {}
    for r in scan_results:
        asset = r.get("asset", "?")
        if asset not in by_asset:
            by_asset[asset] = []
        by_asset[asset].append(r)

    # Build HTML
    html = _build_html(title, rows, by_asset, ts)

    with open(html_path, "w") as f:
        f.write(html)

    # Also save JSON data
    json_path = html_path.replace(".html", ".json")
    with open(json_path, "w") as f:
        json.dump(scan_results, f, indent=2, default=str)

    print(f"[Report] Generated HTML report: {html_path}")
    return html_path


def _build_html(
    title: str,
    rows: list[dict],
    by_asset: dict[str, list[dict]],
    timestamp: str,
) -> str:
    """Build HTML string for the report."""

    # Table rows
    table_rows = ""
    for row in rows:
        cells = "".join(f"<td>{v}</td>" for v in row.values())
        table_rows += f"<tr>{cells}</tr>\n"

    # Table headers
    headers = ""
    if rows:
        headers = "".join(f"<th>{k}</th>" for k in rows[0].keys())

    # Per-asset rankings
    rankings_html = ""
    for asset, results in sorted(by_asset.items()):
        sorted_r = sorted(results, key=lambda x: x.get("avg_score", float("inf")))
        rankings_html += f"<h3>🏆 {asset} — Top Strategies</h3>\n<ol>\n"
        for i, r in enumerate(sorted_r[:5]):
            rankings_html += (
                f'<li><strong>{r["strategy"]}</strong> — '
                f'CRPS: {r.get("avg_score", "∞"):.4f} '
                f'({r.get("frequency", "?")})</li>\n'
            )
        rankings_html += "</ol>\n"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px; margin: 0 auto; padding: 20px;
            background: #f8f9fa; color: #333;
        }}
        h1 {{ color: #1a1a2e; border-bottom: 3px solid #16213e; padding-bottom: 10px; }}
        h2 {{ color: #16213e; margin-top: 30px; }}
        h3 {{ color: #0f3460; }}
        table {{
            width: 100%; border-collapse: collapse;
            background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.12);
            margin: 15px 0;
        }}
        th {{
            background: #16213e; color: white; padding: 12px 8px;
            text-align: left; font-size: 13px;
        }}
        td {{ padding: 10px 8px; border-bottom: 1px solid #e9ecef; font-size: 13px; }}
        tr:hover {{ background: #f1f3f4; }}
        tr:nth-child(even) {{ background: #f8f9fa; }}
        .meta {{ color: #666; font-size: 12px; margin-top: 5px; }}
        .rankings {{ background: white; padding: 15px; border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12); margin: 15px 0; }}
        ol {{ padding-left: 20px; }}
        li {{ margin: 5px 0; }}
    </style>
</head>
<body>
    <h1>📊 {title}</h1>
    <p class="meta">Generated: {timestamp} UTC</p>

    <div class="rankings">
        <h2>Rankings by Asset</h2>
        {rankings_html}
    </div>

    <h2>Full Results</h2>
    <table>
        <thead><tr>{headers}</tr></thead>
        <tbody>{table_rows}</tbody>
    </table>
</body>
</html>"""
