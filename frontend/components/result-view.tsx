import Link from "next/link";
import { Disclaimer } from "@/components/disclaimer";
import { Reveal } from "@/components/motion";
import { MetricCard } from "@/components/metric-card";
import { OcclusionMap } from "@/components/occlusion-map";
import { PlotlyChart } from "@/components/plotly-chart";
import { StatusPill } from "@/components/status-pill";
import { featureText, formatDecimal, formatPercent, labelText } from "@/lib/format";
import type { PredictionResult } from "@/lib/types";

function ProbabilityDial({ probability, label }: { probability: number; label: string }) {
  const circumference = 2 * Math.PI * 54;
  const dash = circumference * probability;
  const color = label === "higher_concern" ? "#0007cb" : label === "uncertain" ? "#4d5757" : "#222f30";
  return (
    <div className="relative grid size-40 place-items-center rounded-full">
      <svg aria-hidden="true" className="absolute inset-0 size-full -rotate-90" viewBox="0 0 120 120">
        <circle cx="60" cy="60" fill="none" r="54" stroke="#dedbd6" strokeWidth="4" />
        <circle cx="60" cy="60" fill="none" r="54" stroke={color} strokeDasharray={`${dash} ${circumference - dash}`} strokeLinecap="butt" strokeWidth="4" />
      </svg>
      <div className="grid size-32 place-items-center rounded-full bg-white text-center">
        <div>
          <p className="text-3xl tracking-[-0.05em] text-ink">{formatPercent(probability)}</p>
          <p className="mt-1 font-mono text-[10px] uppercase tracking-[0.14em] text-graphite">signal</p>
        </div>
      </div>
    </div>
  );
}

function formatFeatureValue(feature: string, value: number): string {
  if (feature === "duration_s") {
    return `${formatDecimal(value, 1)} s`;
  }
  if (feature === "path_length" || feature === "mean_speed" || feature === "pen_lifts") {
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
      marker: { color: contributions.map((item) => (item.value >= 0 ? "#0b5d58" : "#0007cb")) },
      hovertemplate: "%{y}: %{x:+.2f}<extra></extra>",
    },
  ];
  const layout = {
    margin: { l: 8, r: 12, t: 10, b: 30 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    xaxis: { title: "Contribution", gridcolor: "#dedbd6", zerolinecolor: "#c9cbbe", tickfont: { color: "#585858", size: 10 } },
    yaxis: { autorange: "reversed", tickfont: { color: "#222f30", size: 11 }, showgrid: false },
    font: { family: "Inter, Segoe UI, sans-serif", color: "#222f30" },
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
  const modelDescription = result.modelVersion === "demo-preview" ? "Development preview" : "Model service response";

  return (
    <Reveal>
      <div className="mx-auto max-w-7xl px-5 py-12 lg:px-8 lg:py-16">
        <section className="mb-10 flex flex-col gap-8 rounded-[4px] bg-abyss p-6 text-paper sm:p-9 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-lime-100">Review / {new Date(result.createdAt).toLocaleDateString()}</p>
            <h1 className="mt-4 text-5xl leading-[0.95] tracking-[-0.06em] text-white sm:text-6xl">Your movement signal.</h1>
          </div>
          <Link className="rounded-[4px] border border-paper/50 px-4 py-2.5 font-mono text-[10px] uppercase tracking-[0.1em] text-paper transition-colors hover:border-paper hover:bg-paper hover:text-abyss" href="/draw">Make another trace <span aria-hidden="true">↗</span></Link>
        </section>

      <div className="grid gap-5 lg:grid-cols-[0.95fr_1.05fr]">
        <section className="panel p-6 sm:p-8">
          <div className="flex flex-col items-center text-center">
            <StatusPill label={result.label} text={labelText(result.label)} />
            <div className="mt-7"><ProbabilityDial probability={result.probability} label={result.label} /></div>
            <h2 className="mt-7 font-display text-2xl font-semibold text-ink">{labelText(result.label)}</h2>
            <p className="mt-3 max-w-md text-sm leading-6 text-graphite">The model signal sits {formatPercent(Math.abs(result.probability - result.threshold))} from the current threshold of {formatPercent(result.threshold)}. Values within {formatPercent(result.uncertainMargin)} are shown as uncertain.</p>
          </div>
          <div className="mt-8 grid grid-cols-2 gap-3 border-t border-hairline pt-6 text-center">
            <div><p className="text-xs font-semibold uppercase tracking-[0.14em] text-graphite">Model version</p><p className="mt-2 text-sm font-semibold text-ink">{result.modelVersion}</p></div>
            <div><p className="text-xs font-semibold uppercase tracking-[0.14em] text-graphite">Trace type</p><p className="mt-2 text-sm font-semibold capitalize text-ink">{result.kind}</p></div>
          </div>
        </section>

        <section className="panel p-6 sm:p-8">
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="eyebrow">Model sensitivity</p>
              <h2 className="mt-3 font-display text-2xl font-semibold text-ink">Occlusion map</h2>
            </div>
            <span className="rounded-[4px] bg-linen px-3 py-1.5 font-mono text-[10px] uppercase tracking-[0.12em] text-graphite">Research view</span>
          </div>
          <p className="mt-3 max-w-lg text-sm leading-6 text-graphite">Areas that change the model response when covered. This is a model sensitivity view and should not be read as a medical finding.</p>
          <div className="mt-7"><OcclusionMap values={result.occlusionGrid} size={result.occlusionGridSize} /></div>
        </section>
      </div>

      <section className="mt-5">
        <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
          <div>
            <p className="eyebrow">Movement context</p>
            <h2 className="mt-2 font-display text-2xl font-semibold text-ink">What was measured</h2>
          </div>
          <span className="text-xs text-graphite">Features are descriptive, not population norms.</span>
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
          <span className="text-xs text-graphite">Descriptive values from this trace</span>
        </div>
        {featureEntries.length > 0 ? (
          <div className="mt-6 overflow-x-auto">
            <table className="w-full min-w-[620px] text-left text-sm">
              <thead className="border-b border-hairline text-xs uppercase tracking-[0.14em] text-graphite">
                <tr><th className="pb-3 pr-4 font-bold">Feature</th><th className="pb-3 pr-4 font-bold">Value</th><th className="pb-3 font-bold">What it describes</th></tr>
              </thead>
              <tbody className="divide-y divide-ink/10">
                {featureEntries.slice(0, 10).map(([feature, value]) => (
                  <tr key={feature}>
                    <td className="py-4 pr-4 font-semibold text-ink">{featureText(feature)}</td>
                    <td className="py-4 pr-4 font-semibold text-abyss">{formatFeatureValue(feature, value)}</td>
                    <td className="py-4 leading-6 text-graphite">{featureExplanations[feature] ?? "A measured descriptor of this trace."}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="mt-6 rounded-[4px] bg-linen p-4 text-sm text-graphite">No movement feature table was returned by the current service.</p>
        )}
      </section>

      <div className="mt-5 grid gap-5 lg:grid-cols-[1.15fr_0.85fr]">
        <section className="panel p-6 sm:p-8">
          <p className="eyebrow">Feature view</p>
          <h2 className="mt-2 font-display text-2xl font-semibold text-ink">Contribution direction</h2>
          <p className="mt-3 text-sm leading-6 text-graphite">These values describe the model response for this trace. They are not a score for a person.</p>
          <div className="mt-5 h-64"><ContributionChart result={result} /></div>
        </section>
        <section className="panel p-6 sm:p-8">
          <p className="eyebrow">Review notes</p>
          <h2 className="mt-2 font-display text-2xl font-semibold text-ink">Before you interpret this</h2>
          <ul className="mt-5 space-y-3 text-sm leading-6 text-graphite">
            <li className="flex gap-3"><span className="mt-2 size-1.5 shrink-0 rounded-full bg-lime-500" />Use a consistent device and surface for comparisons.</li>
            <li className="flex gap-3"><span className="mt-2 size-1.5 shrink-0 rounded-full bg-lime-500" />A short or noisy trace can make kinematic features less stable.</li>
            <li className="flex gap-3"><span className="mt-2 size-1.5 shrink-0 rounded-full bg-lime-500" />Discuss concerning changes with a qualified clinician.</li>
          </ul>
          <div className="mt-6 rounded-[4px] bg-linen p-4">
            <p className="text-xs font-bold uppercase tracking-[0.14em] text-graphite">Provenance</p>
            <p className="mt-2 text-sm leading-6 text-ink">{result.provenance}</p>
            <p className="mt-2 text-xs leading-5 text-graphite">{modelDescription}. The backend stores features and scores only, never raw drawings.</p>
          </div>
        </section>
      </div>

      {result.warnings.length > 0 ? (
        <section className="mt-5 rounded-[4px] border border-hairline bg-linen p-6">
          <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-violet">Important notes</p>
          <ul className="mt-3 space-y-2 text-sm leading-6 text-graphite">
            {result.warnings.map((warning) => <li key={warning}>• {warning}</li>)}
          </ul>
        </section>
      ) : null}
        <div className="mt-8"><Disclaimer /></div>
      </div>
    </Reveal>
  );
}
