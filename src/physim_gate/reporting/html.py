"""Self-contained HTML report: one file, inline CSS and JS, no network access needed.

Charts are drawn with positioned HTML elements rather than a scaled SVG, so text stays legible at
every width. Every value shown in a chart is also available in a table.
"""

from __future__ import annotations

import json
import math
from html import escape

from physim_gate.comparison import (
    ComparisonResult,
    Evidence,
    InputRef,
    MetricComparison,
    RunSummary,
    Verdict,
)
from physim_gate.reporting.format import (
    cluster_word,
    direction_label,
    interval,
    interval_kind,
    method_label,
    num,
    percent,
    signed,
    unit_suffix,
)

_VERDICT_ICON = {
    Verdict.PASS: "✓",
    Verdict.FAIL: "✕",
    Verdict.INCONCLUSIVE: "?",
    Verdict.INVALID: "!",
}

_EVIDENCE_TEXT = {
    Evidence.REGRESSION: "regression shown",
    Evidence.NON_INFERIOR: "no regression beyond margin shown",
    Evidence.INCONCLUSIVE: "inconclusive",
    Evidence.INSUFFICIENT_SAMPLES: "too few samples",
    Evidence.NOT_EVALUATED: "not evaluated",
}


def _e(value: object) -> str:
    return escape(str(value), quote=True)


def _nice_step(span: float, count: int) -> float:
    raw = span / max(count, 1)
    power = 10.0 ** math.floor(math.log10(raw))
    for factor in (1, 2, 2.5, 5, 10):
        if raw <= factor * power:
            return factor * power
    return 10 * power


def _scale(
    values: list[float], *, include: tuple[float, ...] = ()
) -> tuple[float, float, list[float]]:
    """A padded domain and clean tick values covering ``values`` and ``include``."""
    points = [*values, *include]
    lo, hi = min(points), max(points)
    if lo == hi:
        pad = abs(lo) * 0.1 or 1.0
        lo, hi = lo - pad, hi + pad
    step = _nice_step(hi - lo, 4)
    tick_lo = math.floor(lo / step) * step
    tick_hi = math.ceil(hi / step) * step
    ticks = []
    tick = tick_lo
    while tick <= tick_hi + step / 1e6:
        ticks.append(0.0 if abs(tick) < step / 1e9 else tick)
        tick += step
    return tick_lo, tick_hi, ticks


def _pos(value: float, lo: float, hi: float) -> str:
    """Horizontal position inside a plot, inset so marks on the domain edge are not clipped."""
    return f"{2 + (value - lo) / (hi - lo) * 96:.3f}%"


def _axis(lo: float, hi: float, ticks: list[float]) -> tuple[str, str]:
    grid = "".join(f'<span class="grid" style="left:{_pos(t, lo, hi)}"></span>' for t in ticks)
    labels = "".join(
        f'<span class="tick" style="left:{_pos(t, lo, hi)}">{_e(num(t, 3))}</span>' for t in ticks
    )
    return grid, labels


def _verdict_badge(verdict: Verdict, *, blocking: bool = True, large: bool = False) -> str:
    cls = (
        f"badge v-{verdict.value.lower()}"
        + ("" if blocking else " muted")
        + (" large" if large else "")
    )
    return (
        f'<span class="{cls}"><span class="icon" aria-hidden="true">{_VERDICT_ICON[verdict]}</span>'
        f"{_e(verdict.value)}</span>"
    )


def _is_binary(values: list[float]) -> bool:
    return all(v in (0.0, 1.0) for v in values)


def _rates(metric: MetricComparison) -> str:
    """Binary metrics: one bar per run showing the share of trials with value 1."""
    b, c = metric.baseline, metric.candidate
    assert b is not None
    assert c is not None
    shares = [(sum(s.values) / len(s.values), int(sum(s.values)), len(s.values)) for s in (b, c)]
    _, hi, ticks = _scale([0.0, *(r for r, _, _ in shares)])
    hi = min(hi, 1.0) if max(r for r, _, _ in shares) <= 1.0 else hi
    ticks = [t for t in ticks if t <= hi + 1e-12]
    grid = "".join(f'<span class="grid" style="left:{_bar_pos(t, hi)}"></span>' for t in ticks)
    labels = "".join(
        f'<span class="tick" style="left:{_bar_pos(t, hi)}">{_e(f"{t * 100:.4g}%")}</span>'
        for t in ticks
    )
    lanes = []
    for key, label, (rate, ones, total) in (
        ("base", "Baseline", shares[0]),
        ("cand", "Candidate", shares[1]),
    ):
        text = f"{rate * 100:.1f}% ({ones} of {total})"
        lanes.append(
            f'<div class="lane"><span class="lane-label">{label}</span>'
            f'<div class="lane-plot bars">{grid}'
            f'<span class="bar {key}" style="width:{_bar_pos(rate, hi)}" data-tip="{_e(label)}: {_e(text)} trials with value 1"></span>'
            f'<span class="bar-label" style="left:{_bar_pos(rate, hi)}">{_e(text)}</span>'
            "</div></div>"
        )
    return (
        '<figure class="chart">'
        '<figcaption><span class="chart-title">Share of trials with value 1</span></figcaption>'
        f"{''.join(lanes)}"
        f'<div class="lane axis-lane"><span class="lane-label"></span><div class="lane-plot axis">{labels}</div></div>'
        "</figure>"
    )


def _bar_pos(value: float, hi: float) -> str:
    return f"{value / hi * 80:.3f}%" if hi > 0 else "0%"


def _distribution(metric: MetricComparison) -> str:
    b, c = metric.baseline, metric.candidate
    if b is None or c is None or not b.values or not c.values:
        return ""
    if _is_binary([*b.values, *c.values]):
        return _rates(metric)
    lo, hi, ticks = _scale([*b.values, *c.values])
    grid, labels = _axis(lo, hi, ticks)
    u = unit_suffix(metric)
    stat = metric.statistic.value
    lanes = []
    for key, label, sample in (("base", "Baseline", b), ("cand", "Candidate", c)):
        center = sample.mean if stat == "mean" else sample.median
        dots = []
        for index, (value, trial) in enumerate(zip(sample.values, sample.trial_ids, strict=True)):
            jitter = ((index * 0.6180339887) % 1.0) * 70 + 15
            dots.append(
                f'<span class="dot {key}" style="left:{_pos(value, lo, hi)};top:{jitter:.1f}%" '
                f'data-tip="{_e(label)} · {_e(trial)}: {_e(num(value, 6))}{_e(u)}"></span>'
            )
        center_mark = (
            f'<span class="center {key}" style="left:{_pos(center, lo, hi)}" '
            f'data-tip="{_e(label)} {stat}: {_e(num(center, 6))}{_e(u)}"></span>'
            if center is not None
            else ""
        )
        lanes.append(
            f'<div class="lane"><span class="lane-label">{label}</span>'
            f'<div class="lane-plot">{grid}{"".join(dots)}{center_mark}</div></div>'
        )
    return (
        '<figure class="chart">'
        f'<figcaption><span class="chart-title">Observed values{_e(u and " (" + u.strip() + ")")}</span>'
        '<span class="legend"><span><span class="key base"></span>Baseline</span>'
        '<span><span class="key cand"></span>Candidate</span>'
        f'<span><span class="key center-key"></span>{_e(stat)}</span></span></figcaption>'
        f"{''.join(lanes)}"
        f'<div class="lane axis-lane"><span class="lane-label"></span><div class="lane-plot axis">{labels}</div></div>'
        "</figure>"
    )


def _degradation(metric: MetricComparison) -> str:
    n = metric.numerical
    if n is None:
        return ""
    ci = metric.evidence.degradation_ci
    points = [n.degradation, n.margin, 0.0, *(ci or ())]
    lo, hi, ticks = _scale(points)
    grid, labels = _axis(lo, hi, ticks)
    u = unit_suffix(metric)
    conf = metric.evidence.adjusted_confidence
    band = ""
    if ci is not None:
        band = (
            f'<span class="ci" style="left:{_pos(ci[0], lo, hi)};'
            f'width:calc({_pos(ci[1], lo, hi)} - {_pos(ci[0], lo, hi)})" '
            f'data-tip="{_e(num((conf or 0) * 100, 4))}% interval: {_e(interval(ci))}{_e(u)}"></span>'
        )
    zone_left = _pos(n.margin, lo, hi)
    flip = " flip" if (n.margin - lo) / (hi - lo) > 0.7 else ""
    caption_note = (
        f"{num((conf or 0) * 100, 4)}% {interval_kind(metric)}"
        if ci is not None
        else "no interval (see reasons)"
    )
    return (
        '<figure class="chart">'
        '<figcaption><span class="chart-title">Degradation vs allowed margin</span>'
        '<span class="legend"><span><span class="key point-key"></span>observed</span>'
        f'<span><span class="key ci-key"></span>{_e(caption_note)}</span>'
        '<span><span class="key zone-key"></span>beyond margin</span></span></figcaption>'
        '<div class="lane"><span class="lane-label">Candidate - baseline<small>positive = worse</small></span>'
        f'<div class="lane-plot deg">{grid}'
        f'<span class="zone" style="left:{zone_left};right:0" data-tip="worse than the allowed margin of {_e(num(n.margin))}{_e(u)}"></span>'
        f'<span class="zero" style="left:{_pos(0.0, lo, hi)}"></span>'
        f'<span class="margin-line{flip}" style="left:{zone_left}"><span>margin {_e(num(n.margin))}</span></span>'
        f"{band}"
        f'<span class="point" style="left:{_pos(n.degradation, lo, hi)}" '
        f'data-tip="observed degradation: {_e(signed(n.degradation))}{_e(u)}"></span>'
        "</div></div>"
        f'<div class="lane axis-lane"><span class="lane-label"></span><div class="lane-plot axis">{labels}</div></div>'
        "</figure>"
    )


def _cluster_table(metric: MetricComparison) -> str:
    clusters = metric.clusters
    if clusters is None:
        return ""
    word = cluster_word(metric)
    key = "run" if word == "runs" else clusters.label
    if clusters.mode == "paired":
        rows = "".join(
            f"<tr><td>{_e(k)}</td><td>{_e(num(bv, 6))}</td>"
            f"<td>{_e(num(clusters.candidate_means[k], 6))}</td>"
            f"<td>{_e(signed(clusters.candidate_means[k] - bv, 6))}</td></tr>"
            for k, bv in clusters.baseline_means.items()
        )
        head = (
            f"<tr><th>{_e(key)}</th><th>baseline mean</th><th>candidate mean</th>"
            "<th>difference</th></tr>"
        )
    else:
        b_items = list(clusters.baseline_means.items())
        c_items = list(clusters.candidate_means.items())
        rows = ""
        for i in range(max(len(b_items), len(c_items))):
            cells = [
                f"{_e(items[i][0])}: {_e(num(items[i][1], 6))}" if i < len(items) else ""
                for items in (b_items, c_items)
            ]
            rows += "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
        head = f"<tr><th>baseline ({_e(key)}: mean)</th><th>candidate ({_e(key)}: mean)</th></tr>"
    return (
        f'<details class="data" open><summary>Means of the {_e(word)} compared '
        f"({len(clusters.baseline_means)} baseline, {len(clusters.candidate_means)} candidate)"
        "</summary>"
        f'<div class="scroll"><table class="num"><thead>{head}</thead><tbody>{rows}</tbody></table></div>'
        "</details>"
    )


def _data_table(metric: MetricComparison) -> str:
    b, c = metric.baseline, metric.candidate
    if b is None or c is None:
        return ""
    if metric.pairing is not None and metric.pairing.mode == "paired":
        rows = "".join(
            f"<tr><td>{_e(t)}</td><td>{_e(num(bv, 6))}</td><td>{_e(num(cv, 6))}</td>"
            f"<td>{_e(signed(cv - bv, 6))}</td></tr>"
            for t, bv, cv in zip(b.trial_ids, b.values, c.values, strict=True)
        )
        head = "<tr><th>trial_id</th><th>baseline</th><th>candidate</th><th>difference</th></tr>"
    else:
        width = max(len(b.values), len(c.values))
        rows = ""
        for i in range(width):
            bt = f"{_e(b.trial_ids[i])}: {_e(num(b.values[i], 6))}" if i < len(b.values) else ""
            ct = f"{_e(c.trial_ids[i])}: {_e(num(c.values[i], 6))}" if i < len(c.values) else ""
            rows += f"<tr><td>{bt}</td><td>{ct}</td></tr>"
        head = "<tr><th>baseline (trial: value)</th><th>candidate (trial: value)</th></tr>"
    return (
        f'<details class="data"><summary>Per-trial values ({b.n} baseline, {c.n} candidate)</summary>'
        f'<div class="scroll"><table class="num"><thead>{head}</thead><tbody>{rows}</tbody></table></div>'
        "</details>"
    )


def _facts(metric: MetricComparison) -> str:
    n = metric.numerical
    ev = metric.evidence
    u = unit_suffix(metric).replace(" ", "\u00a0")
    items: list[tuple[str, str]] = []
    if n is not None:
        stat = metric.statistic.value
        items += [
            (f"baseline {stat}", f"{num(n.baseline)}{u}"),
            (f"candidate {stat}", f"{num(n.candidate)}{u}"),
            ("degradation", f"{signed(n.degradation)}{u} ({percent(n.relative_degradation)})"),
            ("allowed margin", f"{num(n.margin)}{u}"),
        ]
    if ev.degradation_ci is not None:
        items.append(
            (
                f"{num((ev.adjusted_confidence or 0) * 100, 4)}% interval",
                f"{interval(ev.degradation_ci)}{u}",
            )
        )
    items.append(("evidence", _EVIDENCE_TEXT[ev.outcome]))
    clusters = metric.clusters
    if clusters is not None:
        word = cluster_word(metric)
        if clusters.mode == "paired":
            sizes = f"{len(clusters.baseline_means)} matched {word}"
        else:
            sizes = f"{len(clusters.baseline_means)} vs {len(clusters.candidate_means)} {word}"
        if word == "clusters":
            sizes += f" ({clusters.label})"
        items.append(("samples", sizes))
    elif metric.pairing is not None:
        if metric.pairing.mode == "paired":
            items.append(("samples", f"{metric.pairing.n_pairs} matched pairs"))
        elif metric.baseline is not None and metric.candidate is not None:
            items.append(("samples", f"{metric.baseline.n} vs {metric.candidate.n}, unpaired"))
    return (
        '<dl class="facts">'
        + "".join(f"<div><dt>{_e(k)}</dt><dd>{_e(v)}</dd></div>" for k, v in items)
        + "</dl>"
    )


def _metric_card(metric: MetricComparison) -> str:
    meta = [f"statistic: {metric.statistic.value}"]
    if metric.unit:
        meta.insert(0, f"unit: {metric.unit}")
    if metric.direction is not None:
        meta.insert(0, direction_label(metric))
    if metric.numerical is not None:
        meta.append(f"margin basis: {metric.numerical.margin_basis}")
    method = method_label(metric)
    if method:
        meta.append(method)
    if not metric.required:
        meta.append("optional (non-blocking)")
    reasons = "".join(f"<li>{_e(r)}</li>" for r in metric.reasons)
    return (
        f'<section class="card metric" id="metric-{_e(metric.name)}">'
        f"<header><h3>{_e(metric.name)}</h3>{_verdict_badge(metric.verdict, blocking=metric.blocking or metric.verdict is Verdict.PASS)}</header>"
        f'<p class="meta">{_e(" · ".join(meta))}</p>'
        f'<ul class="reasons">{reasons}</ul>'
        f"{_facts(metric)}"
        f"{_degradation(metric)}"
        f"{_distribution(metric)}"
        f"{_cluster_table(metric)}"
        f"{_data_table(metric)}"
        "</section>"
    )


def _summary_table(result: ComparisonResult) -> str:
    rows = []
    for m in result.metrics:
        n = m.numerical
        u = unit_suffix(m)
        rows.append(
            "<tr>"
            f'<td><a href="#metric-{_e(m.name)}">{_e(m.name)}</a>'
            f"{'' if m.required else ' <small>(optional)</small>'}</td>"
            f"<td>{_e(num(n.baseline) + u if n else '-')}</td>"
            f"<td>{_e(num(n.candidate) + u if n else '-')}</td>"
            f"<td>{_e(signed(n.degradation) + u if n else '-')}</td>"
            f"<td>{_e(num(n.margin) + u if n else '-')}</td>"
            f"<td>{_e(interval(m.evidence.degradation_ci))}</td>"
            f"<td>{_verdict_badge(m.verdict, blocking=m.blocking or m.verdict is Verdict.PASS)}</td>"
            "</tr>"
        )
    return (
        '<div class="scroll"><table class="num summary"><thead><tr><th>metric</th><th>baseline</th>'
        "<th>candidate</th><th>degradation</th><th>margin</th><th>interval</th><th>verdict</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _run_rows(b: RunSummary, c: RunSummary) -> str:
    def ident(i: object) -> str:
        if i is None:
            return "-"
        version = getattr(i, "version", None)
        return f"{getattr(i, 'id', None) or getattr(i, 'name', '')}" + (
            f" @ {version}" if version else ""
        )

    def env(r: RunSummary) -> str:
        parts = [
            p
            for p in (
                r.environment.python and f"Python {r.environment.python}",
                r.environment.platform,
                r.environment.machine,
            )
            if p
        ]
        parts += [f"{k}={v}" for k, v in r.environment.extra.items()]
        return ", ".join(parts) or "-"

    def src(r: RunSummary) -> str:
        if not r.source.commit:
            return "not recorded"
        text = r.source.commit[:12]
        if r.source.branch:
            text += f" ({r.source.branch})"
        if r.source.dirty:
            text += " +uncommitted changes"
        return text

    fields = [
        ("run_id", b.run_id, c.run_id),
        (
            "created",
            b.created_at.isoformat() if b.created_at else "-",
            c.created_at.isoformat() if c.created_at else "-",
        ),
        ("domain", b.domain.value, c.domain.value),
        ("simulator", ident(b.simulator), ident(c.simulator)),
        ("scenario", ident(b.scenario), ident(c.scenario)),
        ("policy", ident(b.policy), ident(c.policy)),
        ("source", src(b), src(c)),
        ("repository", b.source.repository or "-", c.source.repository or "-"),
        ("environment", env(b), env(c)),
        ("status", b.status.value, c.status.value),
        (
            "trials (not ok)",
            f"{b.n_trials} ({b.n_trials_not_ok})",
            f"{c.n_trials} ({c.n_trials_not_ok})",
        ),
        ("seeds", f"{len(b.seeds)} listed", f"{len(c.seeds)} listed"),
    ]
    return "".join(
        f"<tr><th>{_e(k)}</th><td>{_e(bv)}</td><td{' class=changed' if bv != cv else ''}>{_e(cv)}</td></tr>"
        for k, bv, cv in fields
    )


_CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#6f6d68;
--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--base:#2a78d6;--cand:#eb6834;
--good:#0ca30c;--warn:#fab219;--serious:#ec835a;--crit:#d03b3b;--zone:rgba(208,59,59,.10);--good-ink:#006300;--crit-ink:#b42d2d}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;
--ink2:#c3c2b7;--muted:#9a9890;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--base:#3987e5;--cand:#d95926;
--zone:rgba(208,59,59,.18);--good-ink:#0ca30c;--crit-ink:#e66767}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#9a9890;
--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--base:#3987e5;--cand:#d95926;--zone:rgba(208,59,59,.18);
--good-ink:#0ca30c;--crit-ink:#e66767}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1040px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:15px;font-weight:600;color:var(--ink2);margin:0 0 12px;letter-spacing:.01em}
h2{font-size:18px;margin:40px 0 12px}
h3{font-size:17px;margin:0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;word-break:break-all}
a{color:inherit}
small{color:var(--muted);font-weight:400}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:20px}
.hero{display:flex;flex-direction:column;gap:10px}
.hero .summary{font-size:20px;line-height:1.35;margin:0;font-weight:500}
.hero .scenario{color:var(--ink2);margin:0}
.badge{display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:600;letter-spacing:.04em;
padding:2px 10px 2px 3px;border-radius:999px;border:1px solid var(--ring);white-space:nowrap;color:var(--ink)}
.badge .icon{display:inline-grid;place-items:center;width:18px;height:18px;border-radius:50%;color:#fff;font-size:11px}
.badge.large{font-size:14px;padding:4px 14px 4px 4px;align-self:flex-start}
.badge.large .icon{width:24px;height:24px;font-size:14px}
.v-pass .icon{background:var(--good)}.v-fail .icon{background:var(--crit)}
.v-inconclusive .icon{background:var(--warn);color:#0b0b0b}.v-invalid .icon{background:var(--serious);color:#0b0b0b}
.badge.muted{opacity:.7}.badge.muted::after{content:"non-blocking";font-weight:400;letter-spacing:0;color:var(--muted)}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--grid);vertical-align:top}
thead th{color:var(--muted);font-weight:500;font-size:12px;text-transform:uppercase;letter-spacing:.05em;white-space:nowrap}
table.num td{font-variant-numeric:tabular-nums;white-space:nowrap}
.prov th{width:150px;color:var(--ink2);font-weight:500}
.prov td{word-break:break-word}
.prov td.changed{font-weight:600}
.metric{margin-top:16px}
.metric header{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}
.meta{color:var(--muted);font-size:13px;margin:6px 0 0}
.reasons{margin:12px 0 0;padding-left:18px;color:var(--ink2)}
.reasons li{margin:2px 0}
.facts{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(160px,40%),1fr));gap:12px 16px;margin:16px 0 4px}
.facts div{min-width:0}
.facts dt{color:var(--muted);font-size:12px}
.facts dd{margin:0;font-variant-numeric:tabular-nums;font-weight:500;overflow-wrap:anywhere}
.chart{margin:20px 0 0}
.chart figcaption{display:flex;justify-content:space-between;align-items:baseline;gap:8px 16px;flex-wrap:wrap;margin-bottom:8px}
.chart-title{font-weight:600;font-size:14px}
.legend{display:flex;align-items:center;gap:4px 14px;flex-wrap:wrap;color:var(--ink2);font-size:12px}
.legend>span{display:inline-flex;align-items:center;gap:6px;white-space:nowrap}
.key{display:inline-block;width:9px;height:9px;border-radius:50%}
.key.base{background:var(--base)}.key.cand{background:var(--cand)}
.key.center-key{width:2px;height:12px;border-radius:1px;background:var(--ink)}
.key.point-key{background:var(--ink)}
.key.ci-key{width:16px;height:2px;border-radius:1px;background:var(--ink)}
.key.zone-key{width:12px;height:12px;border-radius:2px;background:var(--zone);border:1px solid var(--crit)}
.lane{display:grid;grid-template-columns:120px 1fr;align-items:center;gap:12px}
.lane-label{font-size:13px;color:var(--ink2);line-height:1.2}
.lane-label small{display:block;font-size:11px}
.lane-plot{position:relative;height:44px}
.lane-plot.deg{height:56px}
.lane-plot.axis{height:20px;border-left:0}
.grid{position:absolute;top:0;bottom:0;width:1px;background:var(--grid)}
.tick{position:absolute;top:2px;transform:translateX(-50%);font-size:11px;color:var(--muted);font-variant-numeric:tabular-nums;white-space:nowrap}
.dot{position:absolute;width:9px;height:9px;margin:-4.5px 0 0 -4.5px;border-radius:50%;box-shadow:0 0 0 1.5px var(--surface);opacity:.85}
.dot.base{background:var(--base)}.dot.cand{background:var(--cand)}
.dot:hover,.center:hover,.point:hover{opacity:1;z-index:3;outline:2px solid var(--ink);outline-offset:1px}
.center{position:absolute;top:4px;bottom:4px;width:2px;margin-left:-1px;background:var(--ink);border-radius:1px;z-index:2}
.center::before{content:"";position:absolute;inset:0 -6px}
.lane-plot.bars{height:40px;border-left:1px solid var(--axis)}
.bar{position:absolute;left:0;top:50%;height:20px;margin-top:-10px;border-radius:0 4px 4px 0}
.bar.base{background:var(--base)}.bar.cand{background:var(--cand)}
.bar-label{position:absolute;top:50%;transform:translateY(-50%);margin-left:8px;font-size:13px;color:var(--ink);white-space:nowrap;font-variant-numeric:tabular-nums}
.zone{position:absolute;top:0;bottom:0;background:var(--zone)}
.zero{position:absolute;top:0;bottom:0;width:1px;background:var(--axis)}
.margin-line{position:absolute;top:0;bottom:0;width:2px;margin-left:-1px;background:var(--crit)}
.margin-line span{position:absolute;top:-2px;left:6px;font-size:11px;color:var(--crit-ink);white-space:nowrap}
.margin-line.flip span{left:auto;right:6px}
.ci{position:absolute;top:calc(50% - 3px);height:12px;border-left:2px solid var(--ink);border-right:2px solid var(--ink);margin-left:-1px}
.ci::before{content:"";position:absolute;left:0;right:0;top:5px;height:2px;background:var(--ink)}
.point{position:absolute;top:calc(50% + 3px);width:12px;height:12px;margin:-6px 0 0 -6px;border-radius:50%;background:var(--ink);box-shadow:0 0 0 2px var(--surface);z-index:2}
details.data{margin-top:16px}
details.data summary{cursor:pointer;color:var(--ink2);font-size:13px}
details.data .scroll{max-height:320px;overflow:auto;margin-top:8px}
.notes li{margin:4px 0}
.blocking{color:var(--crit-ink);font-weight:600}
pre{background:var(--page);border:1px solid var(--ring);border-radius:8px;padding:12px;overflow:auto;font-size:12px;line-height:1.45}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.9em;overflow-wrap:anywhere}
footer{margin-top:40px;color:var(--muted);font-size:13px}
footer p{margin:6px 0}
#tip{position:fixed;z-index:10;pointer-events:none;background:var(--ink);color:var(--surface);font-size:12px;
padding:5px 8px;border-radius:6px;max-width:320px;opacity:0;transition:opacity .08s;font-variant-numeric:tabular-nums}
@media (max-width:640px){.lane{grid-template-columns:76px 1fr;gap:8px}.lane-label{font-size:12px}
.hero .summary{font-size:17px}.card{padding:16px}.prov th{width:100px}}
@media (forced-colors:active){.dot,.point,.center,.ci,.margin-line{forced-color-adjust:none}}
"""

_JS = """
(function(){var tip=document.getElementById('tip');
function show(e){var t=e.target.closest('[data-tip]');if(!t){tip.style.opacity=0;return;}
tip.textContent=t.getAttribute('data-tip');var r=t.getBoundingClientRect();
var x=Math.min(window.innerWidth-tip.offsetWidth-8,Math.max(8,r.left+r.width/2-tip.offsetWidth/2));
var y=r.top-tip.offsetHeight-8;if(y<8)y=r.bottom+8;tip.style.left=x+'px';tip.style.top=y+'px';tip.style.opacity=1;}
document.addEventListener('pointerover',show);document.addEventListener('pointerdown',show);
document.addEventListener('scroll',function(){tip.style.opacity=0;},true);})();
"""


def render_html(result: ComparisonResult) -> str:
    """Render a comparison result as a single self-contained HTML document."""
    scenario = result.candidate.scenario
    notes = "".join(
        f'<li class="blocking">{_e(i)}</li>' for i in result.run_blocking_issues
    ) + "".join(f"<li>{_e(i)}</li>" for i in result.run_issues)
    inputs = result.inputs
    refs: list[tuple[str, InputRef]] = []
    for label, ref, parts in (
        ("baseline", inputs.baseline, inputs.baseline_runs),
        ("candidate", inputs.candidate, inputs.candidate_runs),
    ):
        if parts:
            refs += [(f"{label} {i}/{len(parts)}", part) for i, part in enumerate(parts, 1)]
        else:
            refs.append((label, ref))
    refs.append(("config", inputs.config))
    input_rows = "".join(
        f"<tr><th>{label}</th><td class='mono'>{_e(ref.path or '-')}</td><td class='mono'>{_e(ref.sha256 or '-')}</td></tr>"
        for label, ref in refs
    )
    config_json = json.dumps(result.config.model_dump(mode="json"), indent=2)
    untracked = (
        f"<p class='meta'>Declared but not gated: {_e(', '.join(result.untracked_metrics))}</p>"
        if result.untracked_metrics
        else ""
    )
    title = f"PhySim Gate {result.verdict.value}: {scenario.id}"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="generator" content="physim-gate {_e(result.tool.version)}">
<title>{_e(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<main>
<h1>PhySim Gate regression report</h1>
<section class="card hero">
{_verdict_badge(result.verdict, large=True)}
<p class="summary">{_e(result.summary)}</p>
<p class="scenario">Scenario <strong>{_e(scenario.id)}</strong>{_e(" @ " + scenario.version if scenario.version else "")} · {_e(result.candidate.domain.value.replace("_", " "))} · baseline <span class="mono">{_e(result.baseline.run_id)}</span> → candidate <span class="mono">{_e(result.candidate.run_id)}</span></p>
</section>

<h2>Metrics</h2>
<section class="card">
{_summary_table(result)}
{untracked}
</section>
{"".join(_metric_card(m) for m in result.metrics)}

{f'<h2>Run notes</h2><section class="card"><ul class="notes">{notes}</ul></section>' if notes else ""}

<h2>Provenance</h2>
<section class="card">
<div class="scroll"><table class="prov"><thead><tr><th></th><th>baseline</th><th>candidate</th></tr></thead><tbody>
{_run_rows(result.baseline, result.candidate)}
</tbody></table></div>
<h3 style="font-family:inherit;font-size:14px;margin:20px 0 4px">Inputs</h3>
<div class="scroll"><table><thead><tr><th></th><th>path</th><th>sha256</th></tr></thead><tbody>{input_rows}</tbody></table></div>
<details class="data"><summary>Gate configuration</summary><pre>{_e(config_json)}</pre></details>
</section>

<footer>
<p>How to read this: degradation is candidate minus baseline, signed so that positive is worse. A metric FAILs only when the whole confidence interval lies beyond the allowed margin, and PASSes only when it lies entirely within it. Otherwise the evidence is inconclusive. Intervals are a bootstrap for continuous metrics and a Newcombe score interval for 0/1 metrics, or a t interval over cluster means when trials are grouped into runs or clusters. All are approximate, and the bootstrap can be too narrow with few trials. Matching seeds do not guarantee identical physics. See docs/comparison-semantics.md.</p>
<p>Generated {_e(result.generated_at.isoformat())} by physim-gate {_e(result.tool.version)} · comparison schema {_e(result.schema_version)}</p>
</footer>
</main>
<div id="tip" role="tooltip"></div>
<script>{_JS}</script>
</body>
</html>
"""
