import Link from "next/link";

export function SiteFooter() {
  return (
    <footer className="bg-carbon text-paper">
      <div className="site-container flex flex-col gap-6 py-8 text-sm text-paper/65 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="font-display text-base tracking-[-0.03em] text-white">Kinesis</p>
          <p className="mt-1 font-mono text-[10px] uppercase tracking-[0.14em] text-paper/45">Computer vision for movement traces</p>
        </div>
        <div className="flex flex-wrap gap-x-5 gap-y-2 font-mono text-[10px] uppercase tracking-[0.12em]">
          <Link className="hover:text-lime-100" href="/draw">Open capture</Link>
          <Link className="hover:text-lime-100" href="/history">History</Link>
          <Link className="hover:text-lime-100" href="/about">Method</Link>
        </div>
        <p className="font-mono text-[10px] uppercase tracking-[0.1em] text-paper/45">Research instrument</p>
      </div>
    </footer>
  );
}
