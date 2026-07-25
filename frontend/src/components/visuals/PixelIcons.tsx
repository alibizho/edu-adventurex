type PixelIconProps = {
  className?: string;
};

export function PixelMicIcon({ className }: PixelIconProps) {
  return (
    <svg className={className} viewBox="0 0 48 48" aria-hidden="true" shapeRendering="crispEdges">
      <path d="M18 7h12v4h4v17h-4v4H18v-4h-4V11h4zm2 4v17h8V11zM8 23h5v8h4v4h14v-4h4v-8h5v9h-4v4h-9v6h6v4H15v-4h7v-6h-9v-4H8z" />
    </svg>

  );
}

export function PixelReturnIcon({ className }: PixelIconProps) {
  return (
    <svg className={className} viewBox="0 0 64 64" aria-hidden="true" shapeRendering="crispEdges">
      <path d="M28 6 4 25l24 19V32h16v13H27v11h29V24H28V6z" fill="currentColor" />
    </svg>

  );
}
