import { formatDecimal } from "@/lib/format";
import type { DrawingQuality } from "@/lib/types";

export function QualityMeter({ quality }: { quality: DrawingQuality }) {
  const progress = Math.min((quality.pointCount / 150) * 100, 100);
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between font-mono text-[10px] uppercase tracking-[0.1em]">
        <span className="text-graphite">Trace quality</span>
        <span className={quality.isReady ? "text-abyss" : "text-violet"}>{quality.isReady ? "Ready to review" : "Keep tracing"}</span>
      </div>
      <div aria-label="Trace sample quality" className="h-1.5 overflow-hidden rounded-[4px] bg-hairline" role="progressbar" aria-valuemax={150} aria-valuemin={0} aria-valuenow={quality.pointCount}>
        <div className="h-full rounded-[4px] bg-lime-500 transition-all" style={{ width: `${progress}%` }} />
      </div>
      <div className="flex justify-between font-mono text-[10px] uppercase tracking-[0.08em] text-graphite">
        <span>{quality.pointCount} samples</span>
        <span>{formatDecimal(quality.durationS, 1)} s captured</span>
      </div>
      {quality.warnings.length > 0 ? (
        <ul className="space-y-1 text-xs leading-5 text-violet">
          {quality.warnings.map((warning) => <li key={warning}>• {warning}</li>)}
        </ul>
      ) : (
        <p className="text-xs text-graphite">A continuous trace gives the kinematic analysis more detail.</p>
      )}
      <p className="text-xs leading-5 text-graphite">A full trace is not required to explore the demo, but short traces may produce unstable features.</p>
    </div>
  );
}
