import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ImageUpload } from "./image-upload";

function fileInput(container: HTMLElement): HTMLInputElement {
  const input = container.querySelector<HTMLInputElement>("input[type=file]");
  if (!input) {
    throw new Error("File input was not rendered.");
  }
  return input;
}

describe("ImageUpload", () => {
  it("rejects unsupported file types before analysis", () => {
    const onAnalyze = vi.fn();
    const { container } = render(<ImageUpload isSubmitting={false} kind="spiral" onAnalyze={onAnalyze} />);
    const file = new File(["not an image"], "notes.txt", { type: "text/plain" });
    fireEvent.change(fileInput(container), { target: { files: [file] } });
    expect(screen.getByRole("alert")).toHaveTextContent("Upload a PNG or JPEG image.");
    expect(onAnalyze).not.toHaveBeenCalled();
  });

  it("reads a valid image and passes its data URL to analysis", async () => {
    const onAnalyze = vi.fn().mockResolvedValue(undefined);
    const { container } = render(<ImageUpload isSubmitting={false} kind="wave" onAnalyze={onAnalyze} />);
    const file = new File(["fake png"], "drawing.png", { type: "image/png" });
    fireEvent.change(fileInput(container), { target: { files: [file] } });
    expect(await screen.findByText("drawing.png")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Analyze image" }));
    await waitFor(() => expect(onAnalyze).toHaveBeenCalledWith(expect.stringMatching(/^data:image\/png/)));
  });
});
