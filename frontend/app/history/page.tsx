"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { Disclaimer } from "@/components/disclaimer";
import { Reveal } from "@/components/motion";
import { PlotlyChart } from "@/components/plotly-chart";
import { StatusPill } from "@/components/status-pill";
import { getSessionsFromApi, isMockMode } from "@/lib/api";
import { loadLocalSessions } from "@/lib/client-data";
import { formatDate, formatDecimal, formatPercent, labelText } from "@/lib/format";
import type { SessionRecord } from "@/lib/types";

function mean(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function standardDeviation(values: number[]): number {
  if (values.length < 2) {
    return 0;
  }
  const average = mean(values);
  return Math.sqrt(mean(values.map((value) => (value - average) ** 2)));
}

export default function HistoryPage() {
  const [sessions, setSessions] = useState<SessionRecord[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [apiWarning, setApiWarning] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    if (isMockMode) {
      setSessions(loadLocalSessions());
      setIsLoading(false);
      return () => {
        active = false;
      };
    }
    void getSessionsFromApi()
      .then((apiSessions) => {
        if (active) {
          setSessions(apiSessions ?? loadLocalSessions());
        }
      })
      .catch(() => {
        if (active) {
          setSessions(loadLocalSessions());
          setApiWarning("The history service is unavailable, so this browser's local records are shown.");
        }
      })
      .finally(() => {
        if (active) {
          setIsLoading(false);
        }
      });
    return () => {
      active = false;
    };
  }, []);

  const chronological = useMemo(() => [...sessions].sort((left, right) => left.createdAt.localeCompare(right.createdAt)), [sessions]);
  const baselineValues = chronological.slice(0, 3).map((session) => session.probability);
  const baseline = mean(baselineValues);
  const baselineBand = standardDeviation(baselineValues);
  const chartData = useMemo(() => {
    const x = chronological.map((session) => new Date(session.createdAt));
    const data: Array<Record<string, unknown>> = [
      {
        type: "scatter",
        mode: "lines+markers",
        x,
        y: chronological.map((session) => session.probability),
        name: "Screening signal",
        line: { color: "#0b5d58", width: 3 },
        marker: { color: "#0b5d58", size: 8 },
        hovertemplate: "%{x|%b %d, %Y}<br>%{y:.0%}<extra></extra>",
      },
    ];
    if (baselineValues.length > 1) {
      data.push({
        type: "scatter",
        mode: "lines",
        x,
        y: chronological.map(() => baseline + baselineBand),
        name: "First-session band",
        line: { color: "#4d5757", width: 1, dash: "dot" },
        hovertemplate: "Upper band: %{y:.0%}<extra></extra>",
      });
      data.push({
        type: "scatter",
        mode: "lines",
        x,
        y: chronological.map(() => Math.max(0, baseline - baselineBand)),
        name: "First-session band",
        line: { color: "#4d5757", width: 1, dash: "dot" },
        fill: "tonexty",
        fillcolor: "rgba(201,203,190,0.10)",
        hovertemplate: "Lower band: %{y:.0%}<extra></extra>",
      });
    }
    return data;
  }, [baseline, baselineBand, baselineValues.length, chronological]);

  const featureChartData = useMemo(() => {
    const x = chronological.map((session) => new Date(session.createdAt));
    return [
      {
        type: "scatter",
        mode: "lines+markers",
        x,
        y: chronological.map((session) => session.tremorScore),
        name: "Tremor score",
        yaxis: "y",
        line: { color: "#0b5d58", width: 3 },
        marker: { color: "#0b5d58", size: 7 },
        connectgaps: false,
      },
      {
        type: "scatter",
        mode: "lines+markers",
        x,
        y: chronological.map((session) => session.spiralDeviation),
        name: "Spiral deviation",
        yaxis: "y2",
        line: { color: "#0007cb", width: 3 },
        marker: { color: "#0007cb", size: 7 },
        connectgaps: false,
      },
    ];
  }, [chronological]);

  if (isLoading) {
    return <div className="mx-auto min-h-[60vh] max-w-6xl animate-pulse px-5 py-24"><div className="h-6 w-32 rounded-full bg-stone" /><div className="mt-6 h-16 max-w-xl rounded-[4px] bg-stone" /><div className="mt-8 h-96 rounded-[4px] bg-linen" /></div>;
  }

  return (
    <div className="mx-auto max-w-7xl px-5 py-12 lg:px-8 lg:py-16">
      <Reveal>
        <div className="flex flex-col gap-5 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="eyebrow">Personal context / history</p>
            <h1 className="mt-3 text-5xl leading-[0.95] tracking-[-0.06em] text-ink sm:text-6xl">Your trace history.</h1>
            <p className="mt-4 max-w-2xl text-base leading-7 text-graphite">Compare your own repeated traces on a consistent device. This view does not use population norms or make a clinical comparison.</p>
          </div>
          <Link className="button-primary" href="/draw">Add a trace <span aria-hidden="true">↗</span></Link>
        </div>
      </Reveal>

      {apiWarning ? <p className="mt-6 rounded-[4px] border border-hairline bg-linen px-5 py-4 text-sm text-graphite">{apiWarning}</p> : null}

      {sessions.length === 0 ? (
        <section className="panel mt-10 px-6 py-16 text-center sm:px-10">
          <div className="mx-auto grid size-16 place-items-center rounded-[4px] bg-lime-100 text-3xl text-abyss">↗</div>
          <h2 className="mt-6 font-display text-2xl font-semibold text-ink">Your history starts with one trace.</h2>
          <p className="mx-auto mt-3 max-w-md text-sm leading-6 text-graphite">After you analyze a drawing, its signal and movement features will appear here for personal comparison.</p>
          <Link className="button-primary mt-7" href="/draw">Make the first trace</Link>
        </section>
      ) : (
        <>
          <section className="panel mt-10 p-5 sm:p-8">
            <div className="flex flex-wrap items-end justify-between gap-3">
              <div>
                <p className="eyebrow">Over time</p>
                <h2 className="mt-2 font-display text-2xl font-semibold text-ink">Signal and personal baseline</h2>
              </div>
              <p className="text-xs text-graphite">First-session mean: {formatPercent(baseline)}{baselineValues.length > 1 ? ` ± ${formatPercent(baselineBand)}` : ""}</p>
            </div>
            <div className="mt-6 h-[360px]"><PlotlyChart data={chartData} layout={{ margin: { l: 12, r: 20, t: 10, b: 45 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", yaxis: { title: "Screening signal", range: [0, 1], tickformat: ".0%", gridcolor: "#dedbd6", zerolinecolor: "#c9cbbe" }, xaxis: { type: "date", gridcolor: "#dedbd6" }, font: { family: "Inter, Segoe UI, sans-serif", color: "#222f30" } }} /></div>
          </section>
          <section className="panel mt-5 p-5 sm:p-8">
            <div className="flex flex-wrap items-end justify-between gap-3">
              <div>
                <p className="eyebrow">Movement context</p>
                <h2 className="mt-2 font-display text-2xl font-semibold text-ink">Tremor score and spiral shape</h2>
              </div>
              <p className="text-xs text-graphite">Separate scales are shown for comparison.</p>
            </div>
            <div className="mt-6 h-[360px]"><PlotlyChart data={featureChartData} layout={{ margin: { l: 12, r: 20, t: 10, b: 45 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", yaxis: { title: "Tremor score", gridcolor: "#dedbd6", zerolinecolor: "#c9cbbe" }, yaxis2: { title: "Spiral deviation", overlaying: "y", side: "right", showgrid: false }, xaxis: { type: "date", gridcolor: "#dedbd6" }, font: { family: "Inter, Segoe UI, sans-serif", color: "#222f30" } }} /></div>
          </section>
          <section className="mt-8">
            <div className="mb-4 flex items-end justify-between gap-4">
              <div><p className="eyebrow">Session log</p><h2 className="mt-2 font-display text-2xl font-semibold text-ink">Recent traces</h2></div>
              <Link className="font-mono text-[10px] uppercase tracking-[0.12em] text-violet hover:text-abyss" href="/about">Manage local data ↗</Link>
            </div>
            <div className="grid gap-3">
              {sessions.map((session) => (
                <article className="flex flex-col gap-4 rounded-[4px] border border-hairline bg-white p-5 sm:flex-row sm:items-center sm:justify-between" key={session.id}>
                  <div className="flex items-center gap-4">
                    <div className="grid size-11 place-items-center rounded-[4px] bg-linen text-lg text-ink">{session.kind === "spiral" ? "◎" : "∿"}</div>
                    <div><p className="font-display text-lg font-semibold capitalize text-ink">{session.kind} trace</p><p className="mt-1 text-xs text-graphite">{formatDate(session.createdAt)} · {session.pointerType} input</p></div>
                  </div>
                  <div className="flex flex-wrap items-center gap-4 sm:justify-end">
                    <div className="text-left sm:text-right"><p className="text-xs font-semibold uppercase tracking-[0.14em] text-graphite">Signal</p><p className="mt-1 font-display text-xl font-semibold text-ink">{formatPercent(session.probability)}</p></div>
                    <div className="text-left sm:text-right"><p className="text-xs font-semibold uppercase tracking-[0.14em] text-graphite">Spiral deviation</p><p className="mt-1 text-sm font-semibold text-ink">{session.spiralDeviation === null ? "Not available" : formatDecimal(session.spiralDeviation, 3)}</p></div>
                    <StatusPill label={session.label} text={labelText(session.label)} />
                  </div>
                </article>
              ))}
            </div>
          </section>
        </>
      )}
      <div className="mt-10"><Disclaimer /></div>
    </div>
  );
}
