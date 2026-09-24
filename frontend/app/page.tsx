import Link from "next/link";
import { Disclaimer } from "@/components/disclaimer";
import { SectionHeading } from "@/components/section-heading";

const steps = [
  { number: "01", title: "Choose a template", text: "Select a spiral or wave and use a consistent surface." },
  { number: "02", title: "Trace naturally", text: "Capture position, timing, pressure, and pointer type as you move." },
  { number: "03", title: "Review carefully", text: "See the model signal, movement features, and uncertainty together." },
];

export default function HomePage() {
  return (
    <div>
      <section className="relative overflow-hidden border-b border-ink/10">
        <div className="mx-auto grid max-w-7xl gap-14 px-5 py-20 lg:grid-cols-[1.05fr_0.95fr] lg:items-center lg:px-8 lg:py-28">
          <div className="max-w-2xl">
            <p className="eyebrow">Movement screening / research prototype</p>
            <h1 className="mt-5 max-w-xl font-display text-5xl font-semibold leading-[1.04] tracking-[-0.04em] text-ink sm:text-6xl">A quieter way to notice a change in movement.</h1>
            <p className="mt-7 max-w-xl text-lg leading-8 text-muted">Trace / Signal combines a small drawing model with transparent movement features. It is designed to support reflection, not replace a clinician.</p>
            <div className="mt-9 flex flex-wrap gap-3">
              <Link className="button-primary" href="/draw">Start a trace <span aria-hidden="true">→</span></Link>
              <Link className="button-secondary" href="/about">How it works</Link>
            </div>
            <div className="mt-8"><Disclaimer compact /></div>
          </div>
          <div className="relative mx-auto w-full max-w-[520px]">
            <div className="absolute -inset-5 rounded-[3rem] bg-teal-100/50 blur-2xl" />
            <div className="panel relative overflow-hidden p-5 sm:p-7">
              <div className="flex items-center justify-between border-b border-ink/10 pb-5">
                <div>
                  <p className="eyebrow">Live trace preview</p>
                  <p className="mt-2 font-display text-xl font-semibold text-ink">A sample movement profile</p>
                </div>
                <span className="rounded-full bg-teal-50 px-3 py-1.5 text-xs font-bold text-teal-700">Private by default</span>
              </div>
              <div className="mt-6 grid gap-3 sm:grid-cols-3">
                {["Screening signal", "Tremor features", "Your context"].map((item, index) => (
                  <div className="rounded-2xl bg-paper p-4" key={item}>
                    <div className={`mb-5 h-1.5 w-10 rounded-full ${index === 0 ? "bg-teal-500" : index === 1 ? "bg-amber-500" : "bg-coral-500"}`} />
                    <p className="text-sm font-semibold text-ink">{item}</p>
                    <p className="mt-2 text-xs leading-5 text-muted">Clear, local, and explainable at a glance.</p>
                  </div>
                ))}
              </div>
              <div className="mt-5 rounded-2xl border border-ink/10 bg-ink p-5 text-paper">
                <div className="flex items-end justify-between gap-4">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.18em] text-paper/60">Next step</p>
                    <p className="mt-2 text-xl font-semibold">Make one continuous trace</p>
                  </div>
                  <span className="grid size-12 place-items-center rounded-2xl bg-teal-500/20 text-2xl text-teal-100">↗</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-5 py-20 lg:px-8 lg:py-28">
        <SectionHeading eyebrow="A simple loop" title="Three steps, no black box." description="The interface keeps the model output beside the movement evidence so the result can be understood in context." />
        <div className="mt-12 grid gap-4 md:grid-cols-3">
          {steps.map((step) => (
            <article className="rounded-3xl border border-ink/10 bg-white p-6 shadow-soft" key={step.number}>
              <p className="font-display text-sm font-bold text-teal-600">{step.number}</p>
              <h3 className="mt-8 font-display text-xl font-semibold text-ink">{step.title}</h3>
              <p className="mt-3 text-sm leading-6 text-muted">{step.text}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="border-y border-ink/10 bg-white/50">
        <div className="mx-auto grid max-w-7xl gap-12 px-5 py-20 lg:grid-cols-[0.8fr_1.2fr] lg:items-center lg:px-8 lg:py-24">
          <div>
            <SectionHeading eyebrow="What stays visible" title="A result should help you ask the next question." />
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="rounded-3xl bg-teal-600 p-6 text-white">
              <p className="text-3xl">01</p>
              <h3 className="mt-10 font-display text-xl font-semibold">A plain-language signal</h3>
              <p className="mt-3 text-sm leading-6 text-white/75">The result uses lower concern, higher concern, and uncertain rather than presenting a medical conclusion.</p>
            </div>
            <div className="rounded-3xl bg-ink p-6 text-paper">
              <p className="text-3xl text-teal-200">02</p>
              <h3 className="mt-10 font-display text-xl font-semibold">Movement context</h3>
              <p className="mt-3 text-sm leading-6 text-paper/65">Timing, speed variation, pauses, and spiral shape help describe what the trace contains.</p>
            </div>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-4xl px-5 py-20 text-center lg:py-28">
        <p className="eyebrow">Ready to explore?</p>
        <h2 className="mt-4 font-display text-4xl font-semibold tracking-tight text-ink">Make the trace yours.</h2>
        <p className="mx-auto mt-4 max-w-xl text-base leading-7 text-muted">The current build runs in a clearly labelled development mode until the model service is connected.</p>
        <Link className="button-primary mt-8" href="/draw">Open the drawing studio <span aria-hidden="true">→</span></Link>
      </section>
    </div>
  );
}
