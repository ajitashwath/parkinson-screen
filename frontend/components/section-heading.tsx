interface SectionHeadingProps {
  eyebrow: string;
  title: string;
  description?: string;
  align?: "left" | "center";
}

export function SectionHeading({ eyebrow, title, description, align = "left" }: SectionHeadingProps) {
  const alignment = align === "center" ? "mx-auto text-center" : "";
  return (
    <div className={alignment}>
      <p className="eyebrow">{eyebrow}</p>
      <h2 className="mt-4 max-w-2xl text-4xl leading-[1.02] tracking-[-0.055em] text-ink sm:text-5xl">{title}</h2>
      {description ? <p className="mt-5 max-w-xl text-base leading-7 text-graphite">{description}</p> : null}
    </div>
  );
}
