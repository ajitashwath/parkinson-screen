"use client";

import dynamic from "next/dynamic";
import type { ComponentType } from "react";

type PlotlyProps = {
  data: ReadonlyArray<Record<string, unknown>>;
  layout?: Record<string, unknown>;
  config?: Record<string, unknown>;
  useResizeHandler?: boolean;
};

const Plotly = dynamic(() => import("react-plotly.js"), {
  ssr: false,
  loading: () => <div className="h-64 animate-pulse rounded-2xl bg-ink/5" />,
}) as unknown as ComponentType<PlotlyProps>;

export function PlotlyChart({ data, layout, config, useResizeHandler = true }: PlotlyProps) {
  return (
    <Plotly
      config={{ displayModeBar: false, responsive: true, ...(config ?? {}) }}
      data={data}
      layout={layout}
      useResizeHandler={useResizeHandler}
    />
  );
}
