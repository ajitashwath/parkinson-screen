import { DISCLAIMER } from "@/lib/format";

export function Disclaimer({ compact = false, dark = false }: { compact?: boolean; dark?: boolean }) {
  return (
    <aside className={`flex gap-3 rounded-[4px] border ${dark ? "border-graphite/50 bg-graphite/10 text-paper/75" : "border-hairline bg-linen text-graphite"} ${compact ? "px-4 py-3 text-xs" : "px-5 py-4 text-sm"}`}>
      <span aria-hidden="true" className="mt-0.5 shrink-0 font-mono text-xs text-violet">!</span>
      <p className="leading-relaxed">{DISCLAIMER}</p>
    </aside>
  );
}
