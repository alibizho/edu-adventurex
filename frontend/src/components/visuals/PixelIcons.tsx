/**
 * Pixel-art icons used by more than one screen.
 *
 * Icons that belong to a single screen stay in that file — only the ones that were being
 * copy-pasted live here. The mic path in particular existed identically in three components, so a
 * tweak to the artwork had to be made three times or the screens drifted apart.
 */

type PixelIconProps = {
  className?: string;
};

/** Microphone. Used by the classroom, the zoomed conversation and the static demo workspace. */
export function PixelMicIcon({ className }: PixelIconProps) {
  return (
    <svg className={className} viewBox="0 0 48 48" aria-hidden="true" shapeRendering="crispEdges">
      <path d="M18 7h12v4h4v17h-4v4H18v-4h-4V11h4zm2 4v17h8V11zM8 23h5v8h4v4h14v-4h4v-8h5v9h-4v4h-9v6h6v4H15v-4h7v-6h-9v-4H8z" />
    </svg>
  );
}

/** Back-arrow. Used by the classroom's "back to material" card and the session exit scene. */
export function PixelReturnIcon({ className }: PixelIconProps) {
  return (
    <svg className={className} viewBox="0 0 64 64" aria-hidden="true" shapeRendering="crispEdges">
      <path d="M28 6 4 25l24 19V32h16v13H27v11h29V24H28V6z" fill="currentColor" />
    </svg>
  );
}
