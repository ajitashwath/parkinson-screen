import Link from "next/link";
import { Disclaimer } from "@/components/disclaimer";
import { MetricCard } from "@/components/metric-card";
import { OcclusionMap } from "@/components/occlusion-map";
import { PlotlyChart } from "@/components/plotly-chart";
import { StatusPill } from "@/components/status-pill";
import { featureText, formatDecimal, formatPercent, labelText } from "@/lib/format";
import type { PredictionResult } from "@/lib/types";

function ProbabilityDial({ probability, label }: { probability: number; label: string }) {
  const degrees = Math.round(probability * 360);
  const color = label === "higher_concern" ? "#dc6b5c" : label === "uncertain" ? "#d49a24" : "#159c8c";
  return (
    <div className="relative grid size-40 place-items-center rounded-full" style={{ background: `conic-gradient(${color} ${degrees}deg, #e6ecea ${degrees}deg)` }}>
      <div className="grid size-32 place-items-center rounded-full bg-white text-center shadow-inner">
        <div>
          <p className="font-display text-3xl font-semibold tracking-tight text-ink">{formatPercent(probability)}</p>
          <p className="mt-1 text-[10px] font-bold uppercase tracking-[0.16em] text-muted">signal</p>
        </div>
      </div>
    </div>
  );
}

function formatFeatureValue(feature: string, value: number): string {
  if (feature === "duration_s") {
    return `${formatDecimal(value, 1)} s`;
  }
  if (feature === "path_length" || feature === "mean_speed") {
    return formatDecimal(value, 0);
  }
  if (feature === "peak_freq_hz") {
    return `${formatDecimal(value, 1)} Hz`;
  }
  return formatDecimal(value, 3);
}

function ContributionChart({ result }: { result: PredictionResult }) {
  const contributions = result.contributions.length > 0 ? result.contributions : [{ feature: "tremor_band_ratio", value: 0, explanation: "No contribution data returned" }];
  const data = [
    {
      type: "bar",
      orientation: "h",
      x: contributions.map((item) => item.value),
      y: contributions.map((item) => featureText(item.feature)),
      marker: { color: contributions.map((item) => (item.value >= 0 ? "#159c8c" : "#dc6b5c")) },
      hovertemplate: "%{y}: %{x:+.2f}<extra></extra>",
    },
  ];
  const layout = {
    margin: { l: 8, r: 12, t: 10, b: 30 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    xaxis: { title: "Contribution", gridcolor: "#e6ecea", zerolinecolor: "#cbd8d5", tickfont: { color: "#617381", size: 10 } },
    yaxis: { autorange: "reversed", tickfont: { color: "#123047", size: 11 }, showgrid: false },
    font: { family: "Avenir Next, Segoe UI, sans-serif", color: "#123047" },
  };
  return <PlotlyChart data={data} layout={layout} />;
}

export function ResultView({ result }: { result: PredictionResult }) {
  const featureEntries = Object.entries(result.tremorFeatures).filter(([, value]) => typeof value === "number");
  const deviation = result.spiralDeviation;
  const peakFrequency = result.peakFrequencyHz;
  const featureExplanations: Record<string, string> = {
    duration_s: "How long the trace took to complete.",
    path_length: "Total distance covered by the pointer.",
    mean_speed: "Average pointer movement speed.",
    speed_cv: "How much the movement speed varied.",
    jerk_rms: "How quickly the movement direction changed.",
    pen_lifts: "Number of recorded pen-lift markers.",
    pause_ratio: "Share of the trace spent in slower movement.",
    tremor_band_ratio: "Share of movement energy in the 4 to 12 Hz band.",
    peak_freq_hz: "Strongest rhythm found in the movement signal.",
    spectral_entropy: "How broadly movement energy is spread across frequencies.",
    spiral_deviation: "Difference from a fitted spiral shape.",
    pressure_mean: "Average pressure when pressure data is available.",
    pressure_sd: "Variation in pressure when pressure data is available.",
  };
  const modelDescription = result.provenance.includes("demo") ? "Development preview" : "Model service response";

  return (
    <div className="mx-auto max-w-7xl px-5 py-12 lg:px-8 lg:py-16">
      <div className="mb-8 flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="eyebrow">Trace review / {new Date(result.createdAt).toLocaleDateString()}</p>
          <h1 className="mt-3 font-display text-4xl font-semibold tracking-tight text-ink sm:text-5xl">Your movement signal</h1>
        </div>
        <Link className="button-secondary" href="/draw">Make another trace <span aria-hidden="true">→</span></Link>
      </div>

      <div className="grid gap-5 lg:grid-cols-[0.95fr_1.05fr]">
        <section className="panel p-6 sm:p-8">
          <div className="flex flex-col items-center text-center">
            <StatusPill label={result.label} text={labelText(result.label)} />
            <div className="mt-7"><ProbabilityDial probability={result.probability} label={result.label} /></div>
            <h2 className="mt-7 font-display text-2xl font-semibold text-ink">{labelText(result.label)}</h2>
            <p className="mt-3 max-w-md text-sm leading-6 text-muted">The model signal sits {formatPercent(Math.abs(result.probability - result.threshold))} from the current threshold of {formatPercent(result.threshold)}. Values within {formatPercent(result.uncertainMargin)} are shown as uncertain.</p>
          </div>
          <div className="mt-8 grid grid-cols-2 gap-3 border-t border-ink/10 pt-6 text-center">
            <div><p className="text-xs font-semibold uppercase tracking-[0.14em] text-muted">Model version</p><p className="mt-2 text-sm font-semibold text-ink">{result.modelVersion}</p></div>
            <div><p className="text-xs font-semibold uppercase tracking-[0.14em] text-muted">Trace type</p><p className="mt-2 text-sm font-semibold capitalize text-ink">{result.kind}</p></div>
          </div>
        </section>

        <section className="panel p-6 sm:p-8">
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="eyebrow">Model sensitivity</p>
              <h2 className="mt-3 font-display text-2xl font-semibold text-ink">Occlusion map</h2>
            </div>
            <span className="rounded-full bg-ink/5 px-3 py-1.5 text-[10px] font-bold uppercase tracking-[0.14em] text-muted">Research view</span>
          </div>
          <p className="mt-3 max-w-lg text-sm leading-6 text-muted">Areas that change the model response when covered. This is a model sensitivity view and should not be read as a medical finding.</p>
          <div className="mt-7"><OcclusionMap values={result.occlusionGrid} size={result.occlusionGridSize} /></div>
        </section>
      </div>

      <section className="mt-5">
        <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
          <div>
            <p className="eyebrow">Movement context</p>
            <h2 className="mt-2 font-display text-2xl font-semibold text-ink">What was measured</h2>
          </div>
          <span className="text-xs text-muted">Features are descriptive, not population norms.</span>
        </div>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <MetricCard accent="teal" label="Tremor score" value={result.tremorScore === null ? "Not available" : formatDecimal(result.tremorScore)} detail="Relative movement-band energy in this trace." />
          <MetricCard accent="amber" label="Peak frequency" value={peakFrequency === null ? "Not available" : `${formatDecimal(peakFrequency, 1)} Hz`} detail="Strongest rhythm found in the smoothed velocity signal." />
          <MetricCard accent="coral" label="Spiral deviation" value={deviation === null ? "Not available" : formatDecimal(deviation, 3)} detail="How far the trace departs from a fitted spiral shape." />
          <MetricCard label="Feature report" value={featureEntries.length > 0 ? "Available" : "Limited"} detail="Descriptive movement values returned for this trace." />
        </div>
      </section>

      <section className="panel mt-5 p-6 sm:p-8">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <p className="eyebrow">Feature report</p>
            <h2 className="mt-2 font-display text-2xl font-semibold text-ink">Movement features in plain language</h2>
          </div>
          <span className="text-xs text-muted">Descriptive values from this trace</span>
        </div>
        {featureEntries.length > 0 ? (
          <div className="mt-6 overflow-x-auto">
            <table className="w-full min-w-[620px] text-left text-sm">
              <thead className="border-b border-ink/10 text-xs uppercase tracking-[0.14em] text-muted">
                <tr><th className="pb-3 pr-4 font-bold">Feature</th><th className="pb-3 pr-4 font-bold">Value</th><th className="pb-3 font-bold">What it describes</th></tr>
              </thead>
              <tbody className="divide-y divide-ink/10">
                {featureEntries.slice(0, 10).map(([feature, value]) => (
                  <tr key={feature}>
                    <td className="py-4 pr-4 font-semibold text-ink">{featureText(feature)}</td>
                    <td className="py-4 pr-4 font-semibold text-teal-700">{formatFeatureValue(feature, value)}</td>
                    <td className="py-4 leading-6 text-muted">{featureExplanations[feature] ?? "A measured descriptor of this trace."}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="mt-6 rounded-2xl bg-paper p-4 text-sm text-muted">No movement feature table was returned by the current service.</p>
        )}
      </section>

      <div className="mt-5 grid gap-5 lg:grid-cols-[1.15fr_0.85fr]">
        <section className="panel p-6 sm:p-8">
          <p className="eyebrow">Feature view</p>
          <h2 className="mt-2 font-display text-2xl font-semibold text-ink">Contribution direction</h2>
          <p className="mt-3 text-sm leading-6 text-muted">These values describe the model response for this trace. They are not a score for a person.</p>
          <div className="mt-5 h-64"><ContributionChart result={result} /></div>
        </section>
        <section className="panel p-6 sm:p-8">
          <p className="eyebrow">Review notes</p>
          <h2 className="mt-2 font-display text-2xl font-semibold text-ink">Before you interpret this</h2>
          <ul className="mt-5 space-y-3 text-sm leading-6 text-muted">
            <li className="flex gap-3"><span className="mt-2 size-1.5 shrink-0 rounded-full bg-teal-500" />Use a consistent device and surface for comparisons.</li>
            <li className="flex gap-3"><span className="mt-2 size-1.5 shrink-0 rounded-full bg-teal-500" />A short or noisy trace can make kinematic features less stable.</li>
            <li className="flex gap-3"><span className="mt-2 size-1.5 shrink-0 rounded-full bg-teal-500" />Discuss concerning changes with a qualified clinician.</li>
          </ul>
          <div className="mt-6 rounded-2xl bg-paper p-4">
            <p className="text-xs font-bold uppercase tracking-[0.14em] text-muted">Provenance</p>
            <p className="mt-2 text-sm leading-6 text-ink">{result.provenance}</p>
            <p className="mt-2 text-xs leading-5 text-muted">{modelDescription}. The backend stores features and scores only, never raw drawings.</p>
          </div>
        </section>
      </div>

      {result.warnings.length > 0 ? (
        <section className="mt-5 rounded-3xl border border-amber-200 bg-amber-50 p-6">
          <p className="text-xs font-bold uppercase tracking-[0.16em] text-amber-600">Important notes</p>
          <ul className="mt-3 space-y-2 text-sm leading-6 text-amber-600">
            {result.warnings.map((warning) => <li key={warning}>• {warning}</li>)}
          </ul>
        </section>
      ) : null}
      <div className="mt-8"><Disclaimer /></div>
    </div>
  );
}
