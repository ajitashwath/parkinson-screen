import Link from "next/link";
import { Disclaimer } from "@/components/disclaimer";
import { Reveal, Rule, Stagger, StaggerItem } from "@/components/motion";
import { TraceGlyph } from "@/components/trace-glyph";

const inputs = [
  ["01", "Image geometry", "Stroke shape, spacing, and the visual structure of a spiral or wave."],
  ["02", "Pointer kinematics", "Position, timing, pressure, pauses, and pen-lift events when captured live."],
  ["03", "Spectral rhythm", "Movement-band energy and the frequency pattern in the smoothed velocity signal."],
  ["04", "Provenance", "Model version, threshold, warnings, and the limits of the current interpretation."],
];

const paths = [
  { number: "01", label: "Capture live", detail: "Draw on the canvas with a stylus, touchscreen, or mouse.", href: "/draw" },
  { number: "02", label: "Upload an image", detail: "Bring a clear PNG or JPEG scan into the same analysis flow.", href: "/draw?mode=upload" },
];

export default function HomePage() {
  return (
    <div>
      <div className="border-b border-hairline bg-linen">
        <div className="site-container flex min-h-10 items-center justify-between gap-4 font-mono text-[10px] uppercase tracking-[0.14em] text-graphite">
          <span>Kinesis / movement intelligence</span>
          <span className="hidden sm:inline">A computer vision research instrument</span>
          <span>Edition 01</span>
        </div>
      </div>

      <section className="bg-abyss text-paper">
        <div className="site-container grid min-h-[calc(100svh-112px)] gap-12 py-16 lg:grid-cols-[1.1fr_0.9fr] lg:items-center lg:py-20">
          <Reveal>
            <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-lime-100">Computer vision / movement traces</p>
            <h1 className="mt-8 max-w-3xl text-5xl leading-[0.92] tracking-[-0.065em] text-white sm:text-6xl lg:text-[6.25rem]"><span className="block">Read the <span className="underline decoration-lime decoration-[3px] underline-offset-[10px]">shape</span></span><span className="block">of a movement.</span></h1>
            <p className="mt-8 max-w-xl text-base leading-7 text-paper/70 sm:text-lg sm:leading-8">Kinesis is a research instrument for looking closely at spiral and wave drawings, with the image signal and the movement evidence kept in view together.</p>
            <div className="mt-10 flex flex-wrap gap-3">
              <Link className="rounded-[4px] bg-lime-100 px-5 py-3 font-mono text-xs uppercase tracking-[0.1em] text-abyss transition-colors hover:bg-white" href="/draw">Open capture <span aria-hidden="true">↗</span></Link>
              <Link className="rounded-[4px] border border-paper/50 px-5 py-3 font-mono text-xs uppercase tracking-[0.1em] text-paper transition-colors hover:border-paper hover:bg-paper hover:text-abyss" href="/about">Read the method</Link>
            </div>
            <div className="mt-14 max-w-md border-t border-graphite/50 pt-5">
              <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-paper/50">Working principle</p>
              <p className="mt-2 text-sm leading-6 text-paper/70">Make a trace or upload an image. Review the result with its uncertainty, features, and provenance.</p>
            </div>
          </Reveal>
          <Reveal className="relative min-h-[460px] overflow-hidden border border-graphite/40 lg:min-h-[620px]" delay={0.15}>
            <div className="absolute inset-0 p-5 sm:p-8">
              <TraceGlyph />
            </div>
            <div className="absolute left-5 top-5 font-mono text-[10px] uppercase tracking-[0.14em] text-paper/50 sm:left-8 sm:top-8">K / 01</div>
            <div className="absolute bottom-5 right-5 max-w-[150px] text-right font-mono text-[10px] uppercase leading-5 tracking-[0.12em] text-paper/50 sm:bottom-8 sm:right-8">A visual record<br />not a verdict</div>
          </Reveal>
        </div>
      </section>

      <section className="bg-linen">
        <div className="site-container flex min-h-[65svh] items-center py-20 lg:py-28">
          <div className="grid w-full gap-14 lg:grid-cols-[0.8fr_1.2fr] lg:items-center">
            <Reveal>
            <p className="eyebrow">The instrument</p>
            <h2 className="mt-5 max-w-md text-4xl leading-[0.98] tracking-[-0.06em] text-ink sm:text-5xl">Less interface.<br />More <span className="underline decoration-violet decoration-[3px] underline-offset-[10px]">signal</span>.</h2>
          </Reveal>
          <Reveal className="max-w-2xl lg:pt-12" delay={0.12}>
            <p className="text-lg leading-8 text-ink sm:text-xl sm:leading-8">A polished result should not hide the conditions that produced it. Kinesis keeps the raw mode of capture, the model response, the movement features, and the reasons for uncertainty in the same field of view.</p>
            <Rule className="my-8 text-hairline" />
            <p className="text-base leading-7 text-graphite">Use the instrument for exploration and technical evaluation. It is not a clinical conclusion, and it does not turn a single drawing into a person.</p>
          </Reveal>
          </div>
        </div>
      </section>

      <section className="bg-cream">
        <div className="site-container border-x border-hairline">
          <div className="border-b border-hairline py-6">
            <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-graphite">What enters the model</p>
          </div>
          {inputs.map(([number, title, description]) => (
            <div className="grid gap-3 border-b border-hairline py-6 sm:grid-cols-[80px_220px_1fr] sm:items-baseline" key={number}>
              <span className="font-mono text-xs text-violet">{number}</span>
              <span className="text-lg tracking-[-0.03em] text-ink">{title}</span>
              <span className="text-sm leading-6 text-graphite">{description}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="bg-stone">
        <div className="site-container flex min-h-[75svh] items-center py-20 lg:py-28">
          <div className="w-full">
            <Reveal>
            <p className="eyebrow">Two ways in</p>
            <h2 className="mt-5 max-w-2xl text-4xl leading-[0.98] tracking-[-0.06em] text-ink sm:text-5xl">Start with the <span className="underline decoration-violet decoration-[3px] underline-offset-[10px]">source</span> you have.</h2>
          </Reveal>
          <Stagger className="mt-14 border-t border-ink/20">
            {paths.map((path) => (
              <StaggerItem key={path.number}>
                <Link className="group grid gap-4 border-b border-ink/20 py-8 transition-[background-color,padding] hover:bg-cream/60 sm:grid-cols-[80px_240px_1fr_40px] sm:items-center" href={path.href}>
                  <span className="font-mono text-xs text-violet">{path.number}</span>
                  <span className="text-2xl tracking-[-0.04em] text-ink">{path.label}</span>
                  <span className="text-sm leading-6 text-graphite">{path.detail}</span>
                  <span className="text-right text-xl text-ink transition-transform group-hover:translate-x-1">↗</span>
                </Link>
              </StaggerItem>
            ))}
          </Stagger>
          </div>
        </div>
      </section>

      <section className="bg-cream">
        <div className="site-container flex min-h-[55svh] items-center py-20 lg:py-28">
          <div className="grid w-full gap-10 lg:grid-cols-[1fr_0.8fr] lg:items-end">
            <div>
              <p className="eyebrow">A careful next step</p>
              <h2 className="mt-5 max-w-2xl text-4xl leading-[0.98] tracking-[-0.06em] text-ink sm:text-5xl">Observe first.<br />Interpret second.</h2>
            </div>
            <div>
              <p className="text-base leading-7 text-graphite">Kinesis is built for research, teaching, and technical review. Bring a consistent source, make the process visible, and keep a qualified clinician involved in any health-related question.</p>
              <div className="mt-8"><Disclaimer /></div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
