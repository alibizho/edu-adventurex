/**
 * The six students drawn in `public/images/wut-classroom.png`, as data.
 *
 * Positions are percentages of the scene box, not of the art (1536x1024) — the viewport crops the
 * art to `cover` at `center 24%`, so the two are not the same and a marker placed by art
 * coordinates lands high. The front row is the one that shows it: tuned against the whole picture,
 * those three markers come out sitting on the back row's shoulders.
 *
 * The `?` marker floats above each head; the zoom transform brings that seat to centre at 2.2x,
 * matching the pixel-step camera move the lobby already used for its two hardcoded students.
 *
 * Identity is cosmetic — the backend generates one question at a time and knows nothing about
 * seats. The names exist so a raised hand belongs to *someone*.
 */
export type SeatId =
  | "back-left" | "back-mid" | "back-right"
  | "front-left" | "front-mid" | "front-right";

export type ClassroomSeat = {
  id: SeatId;
  name: string;
  /** Where the `?` bubble sits, as CSS percentages within the scene. */
  marker: { left: string; top: string };
  /** Camera move that centres this seat. Tuned against the art, not computed. */
  zoom: string;
};

export const SEATS: readonly ClassroomSeat[] = [
  { id: "back-left",   name: "MILA",  marker: { left: "18%", top: "14%" }, zoom: "translate(26%, 14%) scale(2.2)" },
  { id: "back-mid",    name: "OTTO",  marker: { left: "43%", top: "14%" }, zoom: "translate(2%, 16%) scale(2.2)" },
  { id: "back-right",  name: "SAFA",  marker: { left: "69%", top: "14%" }, zoom: "translate(-24%, 14%) scale(2.2)" },
  { id: "front-left",  name: "REMY",  marker: { left: "18%", top: "37%" }, zoom: "translate(26%, -6%) scale(2.2)" },
  { id: "front-mid",   name: "JUNO",  marker: { left: "43%", top: "36%" }, zoom: "translate(2%, -4%) scale(2.2)" },
  // Drawn asleep with a zZ, head down on the desk — so their marker hangs lower than the row's.
  // Last to be assigned: waking up to ask a question is the joke.
  { id: "front-right", name: "BODHI", marker: { left: "69%", top: "47%" }, zoom: "translate(-24%, -6%) scale(2.2)" },
];

export const SEAT_BY_ID = new Map(SEATS.map((seat) => [seat.id, seat]));

export function seatName(id: SeatId) {
  return SEAT_BY_ID.get(id)?.name ?? "STUDENT";
}
