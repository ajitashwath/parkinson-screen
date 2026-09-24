import Link from "next/link";

export function SiteFooter() {
  return (
    <footer className="border-t border-ink/10 bg-white/60">
      <div className="mx-auto flex max-w-7xl flex-col gap-5 px-5 py-8 text-sm text-muted sm:flex-row sm:items-center sm:justify-between lg:px-8">
        <div>
          <p className="font-display font-semibold text-ink">Trace / Signal</p>
          <p className="mt-1 text-xs">A careful, privacy-conscious movement screening prototype.</p>
        </div>
        <div className="flex flex-wrap gap-x-5 gap-y-2 text-xs font-medium">
          <Link className="hover:text-ink" href="/draw">Start a trace</Link>
          <Link className="hover:text-ink" href="/history">Your history</Link>
          <Link className="hover:text-ink" href="/about">About the model</Link>
        </div>
        <p className="text-xs">For research and education. Not medical advice.</p>
      </div>
    </footer>
  );
}
