import { DISCLAIMER } from "@/lib/format";

export function Disclaimer({ compact = false }: { compact?: boolean }) {
  return (
    <aside className={`flex gap-3 rounded-2xl border border-amber-200/80 bg-amber-50 text-amber-600 ${compact ? "px-4 py-3 text-xs" : "px-5 py-4 text-sm"}`}>
      <span aria-hidden="true" className="mt-0.5 shrink-0 text-base">!</span>
      <p className="leading-relaxed">{DISCLAIMER}</p>
    </aside>
  );
}
