import { clamp } from "@/lib/format";

const mapColors = ["#f1eee9", "#d3cec6", "#cef79e", "#0007cb"];

function mapColor(value: number): string {
  const normalized = clamp(value, 0, 1);
  const index = Math.min(mapColors.length - 1, Math.floor(normalized * mapColors.length));
  return mapColors[index];
}

export function OcclusionMap({ values, size }: { values: number[][]; size: number }) {
  const safeSize = Math.max(size, 1);
  const cells = Array.from({ length: safeSize }, (_, row) => values[row] ?? []);
  return (
    <div className="space-y-4">
      <div
        aria-label="Occlusion sensitivity map"
        className="mx-auto grid aspect-square w-full max-w-[360px] gap-1 rounded-[4px] border border-hairline bg-white p-2"
        role="img"
        style={{ gridTemplateColumns: `repeat(${safeSize}, minmax(0, 1fr))` }}
      >
        {cells.flatMap((row, rowIndex) =>
          Array.from({ length: safeSize }, (_, columnIndex) => {
            const value = row[columnIndex] ?? 0;
            return <span className="aspect-square rounded-[4px]" key={`${rowIndex}-${columnIndex}`} style={{ backgroundColor: mapColor(value) }} />;
          }),
        )}
      </div>
      <div className="flex items-center gap-3 font-mono text-[10px] uppercase tracking-[0.1em] text-graphite">
        <span>Less influence</span>
        <span aria-hidden="true" className="flex h-2 flex-1 gap-1">
          {mapColors.map((color) => <span className="flex-1 rounded-[4px]" key={color} style={{ backgroundColor: color }} />)}
        </span>
        <span>More influence</span>
      </div>
      <p className="text-xs leading-5 text-graphite">This map shows where covering an area changed the model response. It is a model sensitivity view, not a medical image finding.</p>
    </div>
  );
}
