"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { ResultView } from "@/components/result-view";
import { getResult } from "@/lib/api";
import type { PredictionResult } from "@/lib/types";

export default function ResultPage() {
  const params = useParams<{ id: string }>();
  const id = typeof params.id === "string" ? params.id : "";
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setIsLoading(true);
    setError(null);
    void getResult(id)
      .then((value) => {
        if (active) {
          setResult(value);
          if (!value) {
            setError("This result is not available in the current browser.");
          }
        }
      })
      .catch((reason: unknown) => {
        if (active) {
          setError(reason instanceof Error ? reason.message : "The result could not be loaded.");
        }
      })
      .finally(() => {
        if (active) {
          setIsLoading(false);
        }
      });
    return () => {
      active = false;
    };
  }, [id]);

  if (isLoading) {
    return <div className="site-container min-h-[60vh] animate-pulse py-24"><div className="h-6 w-40 rounded-[4px] bg-stone" /><div className="mt-6 h-16 max-w-xl rounded-[4px] bg-linen" /><div className="mt-8 h-80 rounded-[4px] bg-linen" /></div>;
  }

  if (error || !result) {
    return (
      <div className="mx-auto flex min-h-[60vh] max-w-2xl flex-col items-center justify-center px-5 py-24 text-center">
        <p className="eyebrow">Result unavailable</p>
        <h1 className="mt-4 font-display text-4xl font-semibold text-ink">We could not find that review.</h1>
        <p className="mt-4 text-graphite">{error ?? "Make a new trace and the review will appear here."}</p>
        <Link className="button-primary mt-8" href="/draw">Start a new trace</Link>
      </div>
    );
  }

  return <ResultView result={result} />;
}
