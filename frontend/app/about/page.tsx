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
    <div>
      <section className="bg-abyss text-paper">
        <div className="site-container py-20 lg:py-28">
          <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-lime-100">About this prototype / model card</p>
          <h1 className="mt-6 max-w-4xl text-5xl leading-[0.96] tracking-[-0.06em] text-white sm:text-6xl lg:text-7xl">A transparent research instrument.</h1>
          <p className="mt-7 max-w-2xl text-lg leading-8 text-paper/70">Kinesis is a teaching and research prototype for studying spiral and wave drawings alongside simple movement features. It is intended to make the reasoning visible, not to hide it behind a single number.</p>
        </div>
      </section>

      <section className="bg-cream">
        <div className="site-container grid gap-px overflow-hidden border-x border-b border-hairline bg-hairline py-0 lg:grid-cols-[1.1fr_0.9fr]">
          <div className="bg-white p-7 sm:p-10">
            <p className="eyebrow">Model card summary</p>
            <h2 className="mt-4 text-3xl leading-[1.02] tracking-[-0.05em] text-ink">What the signal represents</h2>
            <dl className="mt-8 divide-y divide-hairline">
              <div className="grid gap-2 py-5 sm:grid-cols-[180px_1fr]"><dt className="text-sm text-ink">Input</dt><dd className="text-sm leading-6 text-graphite">A base64 PNG drawing and a spiral or wave label, plus optional pointer samples for feature analysis.</dd></div>
              <div className="grid gap-2 py-5 sm:grid-cols-[180px_1fr]"><dt className="text-sm text-ink">Image output</dt><dd className="text-sm leading-6 text-graphite">A probability, threshold-aware screening label, and an occlusion sensitivity map.</dd></div>
              <div className="grid gap-2 py-5 sm:grid-cols-[180px_1fr]"><dt className="text-sm text-ink">Movement output</dt><dd className="text-sm leading-6 text-graphite">Timing, speed, pauses, spectral, and spiral-shape features with warnings about input quality.</dd></div>
              <div className="grid gap-2 py-5 sm:grid-cols-[180px_1fr]"><dt className="text-sm text-ink">Data handling</dt><dd className="text-sm leading-6 text-graphite">The intended backend stores features and scores, not raw drawings or raw stroke points.</dd></div>
            </dl>
          </div>
          <div className="bg-abyss p-7 text-paper sm:p-10">
            <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-lime-100">Important boundaries</p>
            <h2 className="mt-4 text-3xl leading-[1.02] tracking-[-0.05em]">Keep the context in view.</h2>
            <ul className="mt-8 space-y-4 text-sm leading-6 text-paper/70">
              {limits.map((limit) => <li className="flex gap-3" key={limit}><span className="mt-2 size-1.5 shrink-0 rounded-full bg-lime-100" />{limit}</li>)}
            </ul>
            <div className="mt-10"><Disclaimer dark /></div>
          </div>
        </div>
      </section>

      <section className="bg-linen">
        <div className="site-container grid gap-8 py-16 lg:grid-cols-[0.75fr_1.25fr] lg:py-20">
          <div>
            <p className="eyebrow">Research data</p>
            <h2 className="mt-4 text-3xl leading-[1.02] tracking-[-0.05em] text-ink">Training context</h2>
          </div>
          <div>
            <p className="text-sm leading-7 text-graphite">The research pipeline is designed around HandPD and NewHandPD spiral and wave drawings, with subject-aware fold handling. The browser preview does not upload a drawing until a configured service is connected.</p>
            <p className="mt-4 text-sm leading-7 text-graphite">Any reported model metrics must come from a real evaluation run. This interface does not display invented accuracy, sensitivity, or population comparisons.</p>
          </div>
        </div>
      </section>

      <section className="bg-cream">
        <div className="site-container border-x border-b border-hairline px-7 py-8 sm:px-10">
          <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="eyebrow">Privacy controls</p>
              <h2 className="mt-3 text-2xl tracking-[-0.04em] text-ink">Your browser, your records.</h2>
              <p className="mt-3 max-w-xl text-sm leading-6 text-graphite">This app stores only features and scores needed for the history page, never raw drawings or strokes. You can remove those records at any time. A connected service may apply its own retention policy.</p>
            </div>
            <DeleteDataButton />
          </div>
        </div>
      </section>

      <div className="site-container flex flex-wrap gap-3 py-12">
        <Link className="button-primary" href="/draw">Try the drawing studio <span aria-hidden="true">↗</span></Link>
        <Link className="button-secondary" href="/history">View history</Link>
      </div>
    </div>
  );
}
