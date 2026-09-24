import { formatDecimal } from "@/lib/format";
import type { DrawingQuality } from "@/lib/types";

export function QualityMeter({ quality }: { quality: DrawingQuality }) {
  const progress = Math.min((quality.pointCount / 150) * 100, 100);
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between text-xs font-semibold">
        <span className="text-muted">Trace quality</span>
        <span className={quality.isReady ? "text-teal-700" : "text-amber-600"}>
          {quality.isReady ? "Ready to review" : "Keep tracing"}
        </span>
      </div>
      <div aria-label="Trace sample quality" className="h-2 overflow-hidden rounded-full bg-ink/10" role="progressbar" aria-valuemax={150} aria-valuemin={0} aria-valuenow={quality.pointCount}>
        <div className="h-full rounded-full bg-gradient-to-r from-teal-500 to-teal-700 transition-all" style={{ width: `${progress}%` }} />
      </div>
      <div className="flex justify-between text-xs text-muted">
        <span>{quality.pointCount} samples</span>
        <span>{formatDecimal(quality.durationS, 1)} s captured</span>
      </div>
      {quality.warnings.length > 0 ? (
        <ul className="space-y-1 text-xs leading-5 text-amber-600">
          {quality.warnings.map((warning) => <li key={warning}>• {warning}</li>)}
        </ul>
      ) : (
        <p className="text-xs text-teal-700">A continuous trace gives the kinematic analysis more detail.</p>
      )}
      <p className="text-xs leading-5 text-muted">A full trace is not required to explore the demo, but short traces may produce unstable features.</p>
    </div>
  );
}
