import { statusClass } from "@/lib/format";

export function StatusPill({ label, text }: { label: string; text?: string }) {
  return (
    <span className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-semibold ${statusClass(label)}`}>
      <span aria-hidden="true" className="size-1.5 rounded-full bg-current" />
      {text ?? label.replaceAll("_", " ")}
    </span>
  );
}
