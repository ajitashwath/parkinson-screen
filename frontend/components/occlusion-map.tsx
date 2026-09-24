import { clamp } from "@/lib/format";

function mapColor(value: number): string {
  const normalized = clamp(value, 0, 1);
  const hue = 172 - normalized * 42;
  const lightness = 88 - normalized * 38;
  const saturation = 48 + normalized * 24;
  return `hsl(${hue} ${saturation}% ${lightness}%)`;
}

export function OcclusionMap({ values, size }: { values: number[][]; size: number }) {
  const safeSize = Math.max(size, 1);
  const cells = Array.from({ length: safeSize }, (_, row) => values[row] ?? []);
  return (
    <div className="space-y-4">
      <div
        aria-label="Occlusion sensitivity map"
        className="mx-auto grid aspect-square w-full max-w-[360px] gap-1 rounded-2xl border border-ink/10 bg-white p-2 shadow-soft"
        role="img"
        style={{ gridTemplateColumns: `repeat(${safeSize}, minmax(0, 1fr))` }}
      >
        {cells.flatMap((row, rowIndex) =>
          Array.from({ length: safeSize }, (_, columnIndex) => {
            const value = row[columnIndex] ?? 0;
            return <span className="aspect-square rounded-[3px]" key={`${rowIndex}-${columnIndex}`} style={{ backgroundColor: mapColor(value) }} />;
          }),
        )}
      </div>
      <div className="flex items-center gap-3 text-xs text-muted">
        <span>Less influence</span>
        <span aria-hidden="true" className="h-2 flex-1 rounded-full bg-[linear-gradient(90deg,hsl(172_48%_88%),hsl(130_72%_50%))]" />
        <span>More influence</span>
      </div>
      <p className="text-xs leading-5 text-muted">This map shows where covering an area changed the model response. It is a model sensitivity view, not a medical image finding.</p>
    </div>
  );
}
