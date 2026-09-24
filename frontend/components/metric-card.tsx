interface MetricCardProps {
  label: string;
  value: string;
  detail: string;
  accent?: "teal" | "amber" | "coral" | "ink";
}

const accents: Record<NonNullable<MetricCardProps["accent"]>, string> = {
  teal: "text-abyss",
  amber: "text-graphite",
  coral: "text-violet",
  ink: "text-ink",
};

export function MetricCard({ label, value, detail, accent = "ink" }: MetricCardProps) {
  return (
    <div className="rounded-[4px] border border-hairline bg-white p-5">
      <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-graphite">{label}</p>
      <p className={`mt-3 text-3xl tracking-[-0.05em] ${accents[accent]}`}>{value}</p>
      <p className="mt-2 text-sm leading-5 text-graphite">{detail}</p>
    </div>
  );
}
