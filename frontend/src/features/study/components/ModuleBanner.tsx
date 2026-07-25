import { Info } from "lucide-react";

type ModuleBannerProps = {
  label: string;
  title: string;
};

export function ModuleBanner({ label, title }: ModuleBannerProps) {
  return (
    <div className="module-banner">
      <div className="module-label">{label}</div>

      <h1>{title}</h1>

      <Info size={27} strokeWidth={2.5} />

    </div>

  );
}
