"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Disclaimer } from "@/components/disclaimer";
import { DrawingCanvas } from "@/components/drawing-canvas";
import { QualityMeter } from "@/components/quality-meter";
import { createPrediction, isMockMode } from "@/lib/api";
import { DRAWING_KINDS, type DrawingCapture, type DrawingKind, type DrawingQuality } from "@/lib/types";

const kindDetails: Record<DrawingKind, { title: string; description: string; cue: string }> = {
  spiral: {
    title: "Spiral",
    description: "Follow the path from the center outward at your natural pace.",
    cue: "Try to keep the line continuous from the center to the outside edge.",
  },
  wave: {
    title: "Wave",
    description: "Trace the repeating wave from left to right in one movement.",
    cue: "Use the full width and keep the rhythm as steady as feels natural.",
  },
};

export default function DrawPage() {
  const router = useRouter();
  const [kind, setKind] = useState<DrawingKind>("spiral");
  const [quality, setQuality] = useState<DrawingQuality>({ durationS: 0, isReady: false, pointCount: 0, warnings: [] });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleKindChange = (nextKind: DrawingKind) => {
    setKind(nextKind);
    setError(null);
  };

  const handleSubmit = async (capture: DrawingCapture) => {
    setIsSubmitting(true);
    setError(null);
    try {
      const result = await createPrediction(capture);
      router.push(`/result/${result.id}`);
    } catch (submissionError) {
      setError(submissionError instanceof Error ? submissionError.message : "The trace could not be prepared.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="mx-auto max-w-7xl px-5 py-12 lg:px-8 lg:py-16">
      <div className="mb-10 flex flex-col gap-5 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="eyebrow">Drawing studio</p>
          <h1 className="mt-3 font-display text-4xl font-semibold tracking-tight text-ink sm:text-5xl">Capture one continuous trace.</h1>
          <p className="mt-4 max-w-2xl text-base leading-7 text-muted">Choose a template, then trace it without trying to make the movement perfect. Natural variation is part of the signal.</p>
        </div>
        <div className="rounded-2xl border border-ink/10 bg-white px-4 py-3 text-xs text-muted shadow-soft">
          <p className="font-semibold text-ink">Pointer capture</p>
          <p className="mt-1">Position · timing · pressure</p>
        </div>
      </div>

      <div className="mb-8 grid gap-3 sm:grid-cols-2">
        {DRAWING_KINDS.map((option) => {
          const details = kindDetails[option];
          const active = kind === option;
          return (
            <button className={`rounded-2xl border p-4 text-left transition-all ${active ? "border-teal-500 bg-teal-50 shadow-soft" : "border-ink/10 bg-white hover:border-ink/25"}`} key={option} onClick={() => handleKindChange(option)} type="button">
              <div className="flex items-center justify-between gap-3">
                <span className="font-display text-lg font-semibold text-ink">{details.title}</span>
                <span className={`rounded-full px-2.5 py-1 text-[10px] font-bold uppercase tracking-[0.14em] ${active ? "bg-teal-600 text-white" : "bg-ink/5 text-muted"}`}>{active ? "Selected" : "Use"}</span>
              </div>
              <p className="mt-2 text-sm leading-5 text-muted">{details.description}</p>
            </button>
          );
        })}
      </div>

      {isMockMode ? <div className="mb-6 rounded-2xl border border-teal-200 bg-teal-50 px-5 py-4 text-sm leading-6 text-teal-700"><span className="font-bold">Development mode.</span> Your trace stays in this page while a deterministic demo response is prepared. Connect FastAPI for model results.</div> : null}

      <div className="grid gap-8 lg:grid-cols-[minmax(0,1.35fr)_minmax(280px,0.65fr)] lg:items-start">
        <div className="panel p-4 sm:p-7">
          <DrawingCanvas key={kind} kind={kind} isSubmitting={isSubmitting} onQualityChange={setQuality} onSubmit={handleSubmit} />
        </div>
        <aside className="space-y-4 lg:sticky lg:top-28">
          <div className="rounded-3xl border border-ink/10 bg-white p-6 shadow-soft">
            <p className="eyebrow">For this trace</p>
            <h2 className="mt-3 font-display text-2xl font-semibold text-ink">{kindDetails[kind].cue}</h2>
            <p className="mt-4 text-sm leading-6 text-muted">Keep your wrist and forearm comfortable. Rest your hand if needed, then begin when you are ready.</p>
            <div className="mt-6 border-t border-ink/10 pt-5">
              <QualityMeter quality={quality} />
            </div>
          </div>
          <div className="rounded-3xl border border-ink/10 bg-ink p-6 text-paper">
            <p className="text-xs font-bold uppercase tracking-[0.18em] text-teal-200">A small request</p>
            <p className="mt-3 text-sm leading-6 text-paper/75">Use a touchscreen or stylus when possible. Mouse input adds quantization and movement noise to kinematic features.</p>
          </div>
        </aside>
      </div>
      {error ? <p className="mt-6 rounded-2xl bg-coral-50 px-5 py-4 text-sm font-medium text-coral-600" role="alert">{error}</p> : null}
      <div className="mt-10"><Disclaimer /></div>
    </div>
  );
}
