"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

const links = [
  { href: "/", label: "Overview" },
  { href: "/draw", label: "Draw" },
  { href: "/history", label: "History" },
  { href: "/about", label: "About" },
];

export function SiteHeader() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  return (
    <header className="sticky top-0 z-40 border-b border-ink/10 bg-paper/90 backdrop-blur-xl">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4 lg:px-8">
        <Link className="group flex items-center gap-3" href="/" onClick={() => setOpen(false)}>
          <span className="grid size-10 place-items-center rounded-2xl bg-ink text-paper shadow-soft transition-transform group-hover:rotate-6">
            <span className="size-4 rounded-full border-2 border-teal-100" />
          </span>
          <span>
            <span className="block font-display text-sm font-semibold tracking-tight text-ink">Trace / Signal</span>
            <span className="block text-[10px] font-medium uppercase tracking-[0.22em] text-muted">Movement screening</span>
          </span>
        </Link>
        <nav aria-label="Main navigation" className="hidden items-center gap-1 md:flex">
          {links.map((link) => {
            const active = link.href === "/" ? pathname === "/" : pathname.startsWith(link.href);
            return (
              <Link
                aria-current={active ? "page" : undefined}
                className={`rounded-full px-4 py-2 text-sm font-medium transition-colors ${
                  active ? "bg-ink text-paper" : "text-muted hover:bg-ink/5 hover:text-ink"
                }`}
                href={link.href}
                key={link.href}
              >
                {link.label}
              </Link>
            );
          })}
          <Link className="ml-3 rounded-full bg-teal-600 px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-teal-700" href="/draw">
            Start a trace
          </Link>
        </nav>
        <button
          aria-expanded={open}
          aria-label={open ? "Close navigation" : "Open navigation"}
          className="rounded-xl border border-ink/10 p-2 text-ink md:hidden"
          onClick={() => setOpen((value) => !value)}
          type="button"
        >
          <span className="block h-0.5 w-5 bg-current" />
          <span className="mt-1 block h-0.5 w-5 bg-current" />
          <span className="mt-1 block h-0.5 w-5 bg-current" />
        </button>
      </div>
      {open ? (
        <nav aria-label="Mobile navigation" className="border-t border-ink/10 bg-paper px-5 py-4 md:hidden">
          <div className="flex flex-col gap-1">
            {links.map((link) => (
              <Link
                className="rounded-xl px-4 py-3 text-sm font-medium text-muted hover:bg-ink/5 hover:text-ink"
                href={link.href}
                key={link.href}
                onClick={() => setOpen(false)}
              >
                {link.label}
              </Link>
            ))}
            <Link className="mt-2 rounded-xl bg-teal-600 px-4 py-3 text-center text-sm font-semibold text-white" href="/draw" onClick={() => setOpen(false)}>
              Start a trace
            </Link>
          </div>
        </nav>
      ) : null}
    </header>
  );
}
