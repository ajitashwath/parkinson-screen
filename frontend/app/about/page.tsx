import Link from "next/link";
import { DeleteDataButton } from "@/components/delete-data-button";
import { Disclaimer } from "@/components/disclaimer";

const limits = [
  "The current app uses a development response until the research model service is connected.",
  "Canvas drawings and scanned training images can have different paper, device, and stroke characteristics.",
  "A short or noisy trace can make kinematic features less reliable.",
  "This prototype is not validated for clinical use or population screening.",
];

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-7xl px-5 py-12 lg:px-8 lg:py-16">
      <div className="max-w-3xl">
        <p className="eyebrow">About this prototype</p>
        <h1 className="mt-3 font-display text-4xl font-semibold tracking-tight text-ink sm:text-5xl">A transparent research instrument.</h1>
        <p className="mt-5 text-lg leading-8 text-muted">Trace / Signal is a teaching and research prototype for studying spiral and wave drawings alongside simple movement features. It is intended to make the reasoning visible, not to hide it behind a single number.</p>
      </div>

      <div className="mt-10 grid gap-5 lg:grid-cols-[1.1fr_0.9fr]">
        <section className="panel p-6 sm:p-8">
          <p className="eyebrow">Model card summary</p>
          <h2 className="mt-2 font-display text-2xl font-semibold text-ink">What the signal represents</h2>
          <dl className="mt-6 divide-y divide-ink/10">
            <div className="grid gap-2 py-4 sm:grid-cols-[180px_1fr]"><dt className="text-sm font-semibold text-ink">Input</dt><dd className="text-sm leading-6 text-muted">A base64 PNG drawing and a spiral or wave label, plus optional pointer samples for feature analysis.</dd></div>
            <div className="grid gap-2 py-4 sm:grid-cols-[180px_1fr]"><dt className="text-sm font-semibold text-ink">Image output</dt><dd className="text-sm leading-6 text-muted">A probability, threshold-aware screening label, and an occlusion sensitivity map.</dd></div>
            <div className="grid gap-2 py-4 sm:grid-cols-[180px_1fr]"><dt className="text-sm font-semibold text-ink">Movement output</dt><dd className="text-sm leading-6 text-muted">Timing, speed, pauses, spectral, and spiral-shape features with warnings about input quality.</dd></div>
            <div className="grid gap-2 py-4 sm:grid-cols-[180px_1fr]"><dt className="text-sm font-semibold text-ink">Data handling</dt><dd className="text-sm leading-6 text-muted">The intended backend stores features and scores, not raw drawings or raw stroke points.</dd></div>
          </dl>
        </section>
        <section className="rounded-3xl bg-ink p-6 text-paper sm:p-8">
          <p className="text-xs font-bold uppercase tracking-[0.18em] text-teal-200">Important boundaries</p>
          <h2 className="mt-3 font-display text-2xl font-semibold">Keep the context in view.</h2>
          <ul className="mt-6 space-y-4 text-sm leading-6 text-paper/70">
            {limits.map((limit) => <li className="flex gap-3" key={limit}><span className="mt-2 size-1.5 shrink-0 rounded-full bg-teal-300" />{limit}</li>)}
          </ul>
          <div className="mt-8"><Disclaimer /></div>
        </section>
      </div>

      <section className="mt-5 rounded-3xl border border-ink/10 bg-white p-6 shadow-soft sm:p-8">
        <div className="grid gap-6 lg:grid-cols-[0.75fr_1.25fr]">
          <div>
            <p className="eyebrow">Research data</p>
            <h2 className="mt-2 font-display text-2xl font-semibold text-ink">Training context</h2>
          </div>
          <div>
            <p className="text-sm leading-6 text-muted">The research pipeline is designed around HandPD and NewHandPD spiral and wave drawings, with subject-aware fold handling. The browser preview does not upload a drawing until a configured service is connected.</p>
            <p className="mt-4 text-sm leading-6 text-muted">Any reported model metrics must come from a real evaluation run. This interface does not display invented accuracy, sensitivity, or population comparisons.</p>
          </div>
        </div>
      </section>

      <section className="mt-5 rounded-3xl border border-ink/10 bg-white p-6 shadow-soft sm:p-8">
        <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="eyebrow">Privacy controls</p>
            <h2 className="mt-2 font-display text-2xl font-semibold text-ink">Your browser, your records.</h2>
            <p className="mt-3 max-w-xl text-sm leading-6 text-muted">The local preview stores only result metadata needed for the history page. You can remove those records at any time. A connected service may apply its own retention policy.</p>
          </div>
          <DeleteDataButton />
        </div>
      </section>

      <div className="mt-10 flex flex-wrap gap-3">
        <Link className="button-primary" href="/draw">Try the drawing studio <span aria-hidden="true">→</span></Link>
        <Link className="button-secondary" href="/history">View history</Link>
      </div>
    </div>
  );
}
