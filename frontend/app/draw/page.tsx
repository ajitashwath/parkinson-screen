"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Disclaimer } from "@/components/disclaimer";
import { DrawingCanvas } from "@/components/drawing-canvas";
import { ImageUpload } from "@/components/image-upload";
import { Reveal } from "@/components/motion";
import { QualityMeter } from "@/components/quality-meter";
import { createPrediction, isMockMode } from "@/lib/api";
import { DRAWING_KINDS, type DrawingCapture, type DrawingKind, type DrawingQuality } from "@/lib/types";

type CaptureMode = "draw" | "upload";

const kindDetails: Record<DrawingKind, { title: string; description: string; cue: string }> = {
  spiral: { title: "Spiral", description: "Follow the path from the center outward at your natural pace.", cue: "Try to keep the line continuous from the center to the outside edge." },
  wave: { title: "Wave", description: "Trace the repeating wave from left to right in one movement.", cue: "Use the full width and keep the rhythm as steady as feels natural." },
};

export default function DrawPage() {
  const router = useRouter();
  const [mode, setMode] = useState<CaptureMode>("draw");
  const [kind, setKind] = useState<DrawingKind>("spiral");
  const [quality, setQuality] = useState<DrawingQuality>({ durationS: 0, isReady: false, pointCount: 0, warnings: [] });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (new URLSearchParams(window.location.search).get("mode") === "upload") {
      setMode("upload");
    }
  }, []);

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

  const handleImageAnalyze = async (imageDataUrl: string) => {
    await handleSubmit({ imageDataUrl, kind, points: [] });
  };

  const setCaptureMode = (nextMode: CaptureMode) => {
    setMode(nextMode);
    setError(null);
  };

  return (
    <div className="site-container py-16 lg:py-24">
      <div className="grid gap-8 border-b border-hairline pb-12 lg:grid-cols-[1fr_320px] lg:items-end">
        <div>
          <p className="eyebrow">Capture / 01</p>
          <h1 className="mt-5 max-w-3xl text-5xl leading-[0.98] tracking-[-0.06em] text-ink sm:text-6xl">Bring a movement into focus.</h1>
          <p className="mt-5 max-w-2xl text-base leading-7 text-graphite">Capture a live trace with a pointer or upload a clear image. Both paths use the same image and movement review.</p>
        </div>
        <div className="border-l border-hairline pl-5">
          <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-graphite">Source modes</p>
          <p className="mt-3 text-sm leading-6 text-ink">Live pointer capture · Image upload</p>
          <p className="mt-3 font-mono text-[10px] uppercase tracking-[0.12em] text-violet">PNG / JPEG · 2 MB maximum</p>
        </div>
      </div>

      <div className="mt-8 flex border-b border-hairline" role="tablist" aria-label="Capture source">
        <button aria-selected={mode === "draw"} className={`border-b-2 px-1 py-3 pr-8 font-mono text-[11px] uppercase tracking-[0.12em] ${mode === "draw" ? "border-ink text-ink" : "border-transparent text-graphite"}`} onClick={() => setCaptureMode("draw")} role="tab" type="button">Capture live</button>
        <button aria-selected={mode === "upload"} className={`border-b-2 px-1 py-3 font-mono text-[11px] uppercase tracking-[0.12em] ${mode === "upload" ? "border-ink text-ink" : "border-transparent text-graphite"}`} onClick={() => setCaptureMode("upload")} role="tab" type="button">Upload image</button>
      </div>

      <div className="mt-8 grid gap-3 sm:grid-cols-2">
        {DRAWING_KINDS.map((option) => {
          const details = kindDetails[option];
          const active = kind === option;
          return (
            <button className={`rounded-[4px] border p-5 text-left transition-colors ${active ? "border-ink bg-linen" : "border-hairline bg-white hover:border-ink"}`} key={option} onClick={() => { setKind(option); setError(null); }} type="button">
              <div className="flex items-center justify-between gap-3">
                <span className="text-2xl tracking-[-0.04em] text-ink">{details.title}</span>
                <span className={`rounded-[4px] px-2.5 py-1 font-mono text-[10px] uppercase tracking-[0.1em] ${active ? "bg-carbon text-white" : "bg-linen text-graphite"}`}>{active ? "Selected" : "Use"}</span>
              </div>
              <p className="mt-3 text-sm leading-6 text-graphite">{details.description}</p>
            </button>
          );
        })}
      </div>

      {isMockMode ? <div className="mt-6 border-l-2 border-lime-500 bg-linen px-5 py-4 text-sm leading-6 text-graphite"><span className="font-mono text-[10px] uppercase tracking-[0.14em] text-abyss">Development mode.</span> Your source stays in this page while a deterministic demo response is prepared. Connect FastAPI for model results.</div> : null}

      <div className="mt-8 grid gap-8 lg:grid-cols-[minmax(0,1.35fr)_minmax(280px,0.65fr)] lg:items-start">
        <Reveal key={mode} className="panel p-4 sm:p-7">
          {mode === "draw" ? <DrawingCanvas key={kind} kind={kind} isSubmitting={isSubmitting} onQualityChange={setQuality} onSubmit={handleSubmit} /> : <ImageUpload kind={kind} isSubmitting={isSubmitting} onAnalyze={handleImageAnalyze} />}
        </Reveal>
        <aside className="space-y-4 lg:sticky lg:top-8">
          <div className="rounded-[4px] border border-hairline bg-white p-6">
            <p className="eyebrow">{mode === "draw" ? "Live trace" : "Image review"}</p>
            <h2 className="mt-4 text-2xl leading-[1.05] tracking-[-0.04em] text-ink">{mode === "draw" ? kindDetails[kind].cue : "A clear source gives the model a fair start."}</h2>
            <p className="mt-4 text-sm leading-6 text-graphite">{mode === "draw" ? "Keep your wrist and forearm comfortable. Rest your hand if needed, then begin when you are ready." : "Upload a flat, well-lit image with the full drawing visible. Camera capture is not used for this workflow."}</p>
            {mode === "draw" ? <div className="mt-6 border-t border-hairline pt-5"><QualityMeter quality={quality} /></div> : <div className="mt-6 border-t border-hairline pt-5 font-mono text-[10px] uppercase leading-5 tracking-[0.12em] text-graphite">Image preprocessing<br />Stroke normalization<br />Model sensitivity view</div>}
          </div>
          <div className="rounded-[4px] bg-abyss p-6 text-paper">
            <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-lime-100">A small request</p>
            <p className="mt-4 text-sm leading-6 text-paper/70">Use a touchscreen or stylus when possible. Mouse input adds quantization and movement noise to kinematic features.</p>
          </div>
        </aside>
      </div>
      {error ? <p className="mt-6 border-l-2 border-violet bg-linen px-5 py-4 text-sm text-violet" role="alert">{error}</p> : null}
      <div className="mt-12"><Disclaimer /></div>
    </div>
  );
}
