declare module "react-plotly.js" {
  import type { ComponentType } from "react";

  interface ReactPlotlyProps {
    data: ReadonlyArray<Record<string, unknown>>;
    layout?: Record<string, unknown>;
    config?: Record<string, unknown>;
    useResizeHandler?: boolean;
  }

  const Plotly: ComponentType<ReactPlotlyProps>;
  export default Plotly;
}
