"use client";

import Image from "next/image";
import { useId, useRef, useState, type ChangeEvent, type DragEvent, type KeyboardEvent } from "react";
import type { DrawingKind } from "@/lib/types";

interface ImageUploadProps {
  kind: DrawingKind;
  isSubmitting: boolean;
  onAnalyze: (imageDataUrl: string) => Promise<void>;
}

const MAX_FILE_BYTES = 2 * 1024 * 1024;
const ACCEPTED_TYPES = new Set(["image/png", "image/jpeg"]);

export function ImageUpload({ kind, isSubmitting, onAnalyze }: ImageUploadProps) {
  const inputId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [imageDataUrl, setImageDataUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const readFile = (file: File) => {
    setError(null);
    if (!ACCEPTED_TYPES.has(file.type)) {
      setError("Upload a PNG or JPEG image.");
      return;
    }
    if (file.size > MAX_FILE_BYTES) {
      setError("Keep the image under 2 MB for a reliable upload.");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result !== "string") {
        setError("The image could not be read.");
        return;
      }
      setImageDataUrl(reader.result);
      setFileName(file.name);
    };
    reader.onerror = () => setError("The image could not be read.");
    reader.readAsDataURL(file);
  };

  const handleInput = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      readFile(file);
    }
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) {
      readFile(file);
    }
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      inputRef.current?.click();
    }
  };

  const clearImage = () => {
    setImageDataUrl(null);
    setFileName(null);
    setError(null);
    if (inputRef.current) {
      inputRef.current.value = "";
    }
  };

  return (
    <div className="space-y-4">
      <input accept="image/png,image/jpeg" className="sr-only" id={inputId} onChange={handleInput} ref={inputRef} type="file" />
      {!imageDataUrl ? (
        <div
          aria-describedby={`${inputId}-note`}
          className={`flex min-h-[360px] cursor-pointer flex-col items-center justify-center border border-hairline bg-cream p-8 text-center transition-colors ${isDragging ? "border-violet bg-linen" : "hover:border-ink"}`}
          onClick={() => inputRef.current?.click()}
          onDragEnter={() => setIsDragging(true)}
          onDragLeave={() => setIsDragging(false)}
          onDragOver={(event) => event.preventDefault()}
          onDrop={handleDrop}
          onKeyDown={handleKeyDown}
          role="button"
          tabIndex={0}
        >
          <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-violet">Image input / {kind}</span>
          <p className="mt-5 max-w-md text-2xl leading-[1.05] tracking-[-0.04em] text-ink">Drop a drawing here, or choose a file.</p>
          <p className="mt-4 max-w-sm text-sm leading-6 text-graphite">PNG or JPEG, up to 2 MB. Use a clear scan or photo of a spiral or wave.</p>
          <span className="mt-7 rounded-[4px] bg-carbon px-4 py-2.5 font-mono text-[10px] uppercase tracking-[0.1em] text-white">Choose image</span>
        </div>
      ) : (
        <div className="border border-hairline bg-cream p-4">
          <div className="relative aspect-square w-full overflow-hidden rounded-[4px] border border-hairline bg-linen">
            <Image alt="Uploaded drawing preview" className="object-contain" fill priority sizes="(max-width: 1024px) 90vw, 680px" src={imageDataUrl} unoptimized />
          </div>
          <div className="mt-4 flex flex-col gap-3 border-t border-hairline pt-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="min-w-0">
              <p className="truncate text-sm text-ink">{fileName}</p>
              <p className="mt-1 font-mono text-[10px] uppercase tracking-[0.12em] text-graphite">Ready for image analysis / {kind}</p>
            </div>
            <div className="flex flex-wrap gap-2">
              <button className="rounded-[4px] border border-hairline px-3 py-2 font-mono text-[10px] uppercase tracking-[0.1em] text-graphite hover:border-ink hover:text-ink" onClick={clearImage} type="button">Replace</button>
              <button className="rounded-[4px] bg-carbon px-3 py-2 font-mono text-[10px] uppercase tracking-[0.1em] text-white disabled:opacity-45" disabled={isSubmitting} onClick={() => void onAnalyze(imageDataUrl)} type="button">{isSubmitting ? "Analyzing" : "Analyze image"}</button>
            </div>
          </div>
        </div>
      )}
      <p className="font-mono text-[10px] uppercase tracking-[0.1em] text-graphite" id={`${inputId}-note`}>Camera capture is not required for this workflow.</p>
      {error ? <p className="border-l-2 border-violet bg-linen px-4 py-3 text-sm text-violet" role="alert">{error}</p> : null}
    </div>
  );
}
