interface MetricCardProps {
  label: string;
  value: string;
  detail: string;
  accent?: "teal" | "amber" | "coral" | "ink";
}

const accents: Record<NonNullable<MetricCardProps["accent"]>, string> = {
  teal: "text-teal-600",
  amber: "text-amber-600",
  coral: "text-coral-600",
  ink: "text-ink",
};

export function MetricCard({ label, value, detail, accent = "ink" }: MetricCardProps) {
  return (
    <div className="rounded-2xl border border-ink/10 bg-white p-5 shadow-soft">
      <p className="text-xs font-semibold uppercase tracking-[0.16em] text-muted">{label}</p>
      <p className={`mt-3 font-display text-3xl font-semibold tracking-tight ${accents[accent]}`}>{value}</p>
      <p className="mt-2 text-sm leading-5 text-muted">{detail}</p>
    </div>
  );
}
