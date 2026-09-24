"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import type { DrawingCapture, DrawingKind, DrawingQuality, StrokePoint } from "@/lib/types";

interface DrawingCanvasProps {
  kind: DrawingKind;
  isSubmitting: boolean;
  onSubmit: (capture: DrawingCapture) => Promise<void>;
  onQualityChange?: (quality: DrawingQuality) => void;
}

function drawTemplate(context: CanvasRenderingContext2D, size: number, kind: DrawingKind) {
  context.save();
  context.strokeStyle = "#d7e2e0";
  context.lineWidth = 2;
  context.setLineDash([6, 8]);
  context.beginPath();
  if (kind === "spiral") {
    const center = size / 2;
    const maxRadius = size * 0.39;
    for (let step = 0; step <= 260; step += 1) {
      const progress = step / 260;
      const angle = progress * Math.PI * 7.2;
      const radius = maxRadius * progress;
      const x = center + Math.cos(angle) * radius;
      const y = center + Math.sin(angle) * radius;
      if (step === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    }
  } else {
    const left = size * 0.1;
    const right = size * 0.9;
    for (let step = 0; step <= 220; step += 1) {
      const progress = step / 220;
      const x = left + (right - left) * progress;
      const y = size / 2 + Math.sin(progress * Math.PI * 4) * size * 0.22;
      if (step === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    }
  }
  context.stroke();
  context.restore();
}

function drawStroke(context: CanvasRenderingContext2D, points: StrokePoint[]) {
  if (points.length === 0) {
    return;
  }
  context.save();
  context.strokeStyle = "#172b3a";
  context.fillStyle = "#172b3a";
  context.lineWidth = 6;
  context.lineCap = "round";
  context.lineJoin = "round";
  if (points.length === 1) {
    context.beginPath();
    context.arc(points[0].x, points[0].y, 3, 0, Math.PI * 2);
    context.fill();
    context.restore();
    return;
  }
  context.beginPath();
  context.moveTo(points[0].x, points[0].y);
  for (const point of points.slice(1)) {
    context.lineTo(point.x, point.y);
  }
  context.stroke();
  context.restore();
}

function pointsFromStrokes(strokes: StrokePoint[][], current: StrokePoint[] | null): StrokePoint[] {
  return [...strokes.flat(), ...(current ?? [])];
}

export function DrawingCanvas({ kind, isSubmitting, onSubmit, onQualityChange }: DrawingCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const strokesRef = useRef<StrokePoint[][]>([]);
  const currentStrokeRef = useRef<StrokePoint[] | null>(null);
  const activePointerRef = useRef<number | null>(null);
  const strokeStartedAtRef = useRef(0);
  const sessionStartedAtRef = useRef(0);
  const pixelRatioRef = useRef(1);
  const onQualityChangeRef = useRef(onQualityChange);
  const [canvasSize, setCanvasSize] = useState(600);
  const [strokeVersion, setStrokeVersion] = useState(0);
  const [isDrawing, setIsDrawing] = useState(false);
  const [quality, setQuality] = useState<DrawingQuality>({
    durationS: 0,
    isReady: false,
    pointCount: 0,
    warnings: [],
  });
  const [submitError, setSubmitError] = useState<string | null>(null);

  useEffect(() => {
    onQualityChangeRef.current = onQualityChange;
  }, [onQualityChange]);

  const updateQuality = useCallback(() => {
    const points = pointsFromStrokes(strokesRef.current, currentStrokeRef.current);
    const first = points[0];
    const last = points[points.length - 1];
    const durationS = first && last ? Math.max(0, (last.t_ms - first.t_ms) / 1000) : 0;
    const pointerTypes = new Set(points.map((point) => point.pointer_type));
    const warnings: string[] = [];
    if (points.length < 50) {
      warnings.push("Aim for at least 50 samples for a richer trace.");
    }
    if (durationS < 2) {
      warnings.push("Trace for at least two seconds where possible.");
    }
    if (pointerTypes.has("mouse")) {
      warnings.push("Mouse input can add movement noise. A stylus or touchscreen is preferred.");
    }
    const nextQuality: DrawingQuality = {
      durationS,
      isReady: points.length >= 50 && durationS >= 2,
      pointCount: points.length,
      warnings,
    };
    setQuality(nextQuality);
    onQualityChangeRef.current?.(nextQuality);
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) {
      return;
    }
    const resize = () => {
      const width = Math.max(280, Math.min(720, container.clientWidth || 600));
      const ratio = Math.max(1, window.devicePixelRatio || 1);
      pixelRatioRef.current = ratio;
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(width * ratio);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${width}px`;
      setCanvasSize(width);
    };
    resize();
    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", resize);
      return () => window.removeEventListener("resize", resize);
    }
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }
    const ratio = pixelRatioRef.current;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, canvasSize, canvasSize);
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, canvasSize, canvasSize);
    drawTemplate(context, canvasSize, kind);
    for (const stroke of strokesRef.current) {
      drawStroke(context, stroke);
    }
    if (currentStrokeRef.current) {
      drawStroke(context, currentStrokeRef.current);
    }
  }, [canvasSize, kind, strokeVersion]);

  const addSample = (event: ReactPointerEvent<HTMLCanvasElement> | PointerEvent) => {
    if (!currentStrokeRef.current) {
      return;
    }
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const bounds = canvas.getBoundingClientRect();
    const x = ((event.clientX - bounds.left) / bounds.width) * canvasSize;
    const y = ((event.clientY - bounds.top) / bounds.height) * canvasSize;
    const previous = currentStrokeRef.current[currentStrokeRef.current.length - 1];
    if (previous && previous.x === x && previous.y === y) {
      return;
    }
    const pressure = "pressure" in event && event.pressure > 0 ? event.pressure : 0.5;
    const pointerType = "pointerType" in event ? event.pointerType : "mouse";
    currentStrokeRef.current.push({
      x: Math.round(x * 10) / 10,
      y: Math.round(y * 10) / 10,
      t_ms: Math.max(0, Math.round(performance.now() - (sessionStartedAtRef.current || strokeStartedAtRef.current))),
      pressure,
      pointer_type: pointerType,
      pen_lift: false,
    });
    setStrokeVersion((version) => version + 1);
    updateQuality();
  };

  const beginStroke = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (event.button !== 0 || activePointerRef.current !== null || isSubmitting) {
      return;
    }
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    event.currentTarget.focus();
    activePointerRef.current = event.pointerId;
    if (strokesRef.current.length === 0) {
      sessionStartedAtRef.current = performance.now();
    }
    strokeStartedAtRef.current = performance.now();
    currentStrokeRef.current = [];
    setIsDrawing(true);
    setSubmitError(null);
    addSample(event);
  };

  const moveStroke = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (activePointerRef.current !== event.pointerId || !currentStrokeRef.current) {
      return;
    }
    event.preventDefault();
    const nativeEvent = event.nativeEvent;
    const samples = typeof nativeEvent.getCoalescedEvents === "function" ? nativeEvent.getCoalescedEvents() : [nativeEvent];
    for (const sample of samples) {
      addSample(sample);
    }
  };

  const finishStroke = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (activePointerRef.current !== event.pointerId) {
      return;
    }
    event.preventDefault();
    if (currentStrokeRef.current) {
      addSample(event);
      if (currentStrokeRef.current.length > 0) {
        strokesRef.current.push([...currentStrokeRef.current]);
      }
    }
    currentStrokeRef.current = null;
    activePointerRef.current = null;
    setIsDrawing(false);
    setStrokeVersion((version) => version + 1);
    updateQuality();
  };

  const clear = () => {
    strokesRef.current = [];
    currentStrokeRef.current = null;
    activePointerRef.current = null;
    sessionStartedAtRef.current = 0;
    setIsDrawing(false);
    setStrokeVersion((version) => version + 1);
    setSubmitError(null);
    updateQuality();
  };

  const undo = () => {
    if (currentStrokeRef.current) {
      currentStrokeRef.current = null;
      activePointerRef.current = null;
      setIsDrawing(false);
    } else {
      strokesRef.current.pop();
      if (strokesRef.current.length === 0) {
        sessionStartedAtRef.current = 0;
      }
    }
    setStrokeVersion((version) => version + 1);
    setSubmitError(null);
    updateQuality();
  };

  const submit = async () => {
    const points = pointsFromStrokes(strokesRef.current, currentStrokeRef.current);
    const canvas = canvasRef.current;
    if (points.length === 0 || !canvas) {
      setSubmitError("Trace at least one continuous line before analyzing.");
      return;
    }
    setSubmitError(null);
    try {
      await onSubmit({
        imageDataUrl: canvas.toDataURL("image/png"),
        kind,
        points,
      });
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : "The trace could not be analyzed.");
    }
  };

  const hasPoints = quality.pointCount > 0;

  return (
    <div className="space-y-4">
      <div ref={containerRef} className="relative mx-auto w-full max-w-[680px]">
        <canvas
          aria-describedby="draw-instructions"
          aria-label={`${kind} drawing canvas`}
          className="aspect-square w-full touch-none rounded-3xl border border-ink/10 bg-white shadow-card outline-none ring-offset-4 focus-visible:ring-2 focus-visible:ring-teal-500"
          onPointerCancel={finishStroke}
          onPointerDown={beginStroke}
          onPointerMove={moveStroke}
          onPointerUp={finishStroke}
          ref={canvasRef}
          role="img"
          tabIndex={0}
        />
        {isDrawing ? <div className="pointer-events-none absolute left-4 top-4 rounded-full bg-ink/90 px-3 py-1.5 text-xs font-semibold text-white">Tracing…</div> : null}
      </div>
      <p className="sr-only" id="draw-instructions">Draw a continuous {kind} over the faint template. Press and hold, then trace the shape in one movement.</p>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex gap-2">
          <button className="rounded-full border border-ink/15 px-4 py-2 text-sm font-semibold text-muted transition-colors hover:border-ink/30 hover:text-ink disabled:opacity-40" disabled={!hasPoints || isSubmitting} onClick={undo} type="button">Undo</button>
          <button className="rounded-full border border-ink/15 px-4 py-2 text-sm font-semibold text-muted transition-colors hover:border-ink/30 hover:text-ink disabled:opacity-40" disabled={!hasPoints || isSubmitting} onClick={clear} type="button">Clear</button>
        </div>
        <button className="inline-flex items-center gap-2 rounded-full bg-ink px-5 py-2.5 text-sm font-semibold text-white shadow-soft transition-colors hover:bg-teal-700 disabled:cursor-not-allowed disabled:opacity-45" disabled={!hasPoints || isSubmitting} onClick={() => void submit()} type="button">
          {isSubmitting ? "Preparing review…" : "Analyze trace"}
          <span aria-hidden="true">→</span>
        </button>
      </div>
      {submitError ? <p className="rounded-xl bg-coral-50 px-4 py-3 text-sm font-medium text-coral-600" role="alert">{submitError}</p> : null}
    </div>
  );
}
