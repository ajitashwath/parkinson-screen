"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

const links = [
  { href: "/", label: "Overview" },
  { href: "/draw", label: "Capture" },
  { href: "/history", label: "History" },
  { href: "/about", label: "Method" },
];

export function SiteHeader() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  return (
    <header className="border-b border-hairline bg-cream">
      <div className="site-container flex min-h-[72px] items-center justify-between gap-6">
        <Link className="group" href="/" onClick={() => setOpen(false)}>
          <span className="block text-lg tracking-[-0.04em] text-ink">Kinesis</span>
          <span className="mt-0.5 block font-mono text-[9px] uppercase tracking-[0.16em] text-graphite">Computer vision for movement traces</span>
        </Link>
        <nav aria-label="Main navigation" className="hidden items-center gap-7 md:flex">
          {links.map((link) => {
            const active = link.href === "/" ? pathname === "/" : pathname.startsWith(link.href);
            return (
              <Link
                aria-current={active ? "page" : undefined}
                className={`border-b py-2 font-mono text-[11px] uppercase tracking-[0.12em] transition-colors ${active ? "border-ink text-ink" : "border-transparent text-graphite hover:border-stone hover:text-ink"}`}
                href={link.href}
                key={link.href}
              >
                {link.label}
              </Link>
            );
          })}
          <Link className="rounded-[4px] bg-carbon px-4 py-2.5 font-mono text-[11px] uppercase tracking-[0.1em] text-white transition-colors hover:bg-abyss" href="/draw">
            Open capture
          </Link>
        </nav>
        <button aria-expanded={open} aria-label={open ? "Close navigation" : "Open navigation"} className="rounded-[4px] border border-hairline p-2 text-ink md:hidden" onClick={() => setOpen((value) => !value)} type="button">
          <span className="block h-px w-5 bg-current" />
          <span className="mt-1 block h-px w-5 bg-current" />
          <span className="mt-1 block h-px w-5 bg-current" />
        </button>
      </div>
      {open ? (
        <nav aria-label="Mobile navigation" className="border-t border-hairline bg-linen px-5 py-4 md:hidden">
          <div className="flex flex-col gap-1">
            {links.map((link) => <Link className="rounded-[4px] px-4 py-3 font-mono text-xs uppercase tracking-[0.1em] text-graphite hover:bg-white hover:text-ink" href={link.href} key={link.href} onClick={() => setOpen(false)}>{link.label}</Link>)}
            <Link className="mt-2 rounded-[4px] bg-carbon px-4 py-3 text-center font-mono text-xs uppercase tracking-[0.1em] text-white" href="/draw" onClick={() => setOpen(false)}>Open capture</Link>
          </div>
        </nav>
      ) : null}
    </header>
  );
}
