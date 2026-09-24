import Link from "next/link";

export default function NotFound() {
  return (
    <div className="mx-auto flex min-h-[60vh] max-w-3xl flex-col items-center justify-center px-5 py-24 text-center">
      <p className="eyebrow">Page not found</p>
      <h1 className="mt-4 font-display text-4xl font-semibold text-ink">That page has moved on.</h1>
      <p className="mt-4 text-graphite">The link may be old, or the trace may not be stored in this browser.</p>
      <Link className="button-primary mt-8" href="/">Return home</Link>
    </div>
  );
}
