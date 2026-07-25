/**
 * The six desks in `public/images/classroom.png`, as data.
 *
 * The room is drawn empty and the students are composited onto it, so these are measured off the
 * art itself: `x` is the centre of a desk and `base` is the top edge of the surface the student
 * sits behind. Everything in the scene is positioned in these units and the scene box is given
 * the art's own aspect ratio, so a percentage here is a percentage of the picture — there is no
 * crop to map through, and re-cropping the viewport cannot slide the students off their desks.
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
  /** Centre of the desk, as a percentage of the scene's width. */
  x: number;
  /** Top edge of the desk this student sits behind, as a percentage of the scene's height. */
  base: number;
};

/** The two rows of desks, read off the art. */
const BACK_ROW =41.9;
const FRONT_ROW = 57.5;

export const SEATS: readonly ClassroomSeat[] = [
  { id: "back-left",   name: "MILA",  x: 27.7, base: BACK_ROW },
  { id: "back-mid",    name: "OTTO",  x: 49.8, base: BACK_ROW },
  { id: "back-right",  name: "SAFA",  x: 74.8, base: BACK_ROW },
  { id: "front-left",  name: "REMY",  x: 25.7, base: FRONT_ROW },
  { id: "front-mid",   name: "JUNO",  x: 50.8, base: FRONT_ROW },
  { id: "front-right", name: "BODHI", x: 75.8, base: FRONT_ROW },
];

export const SEAT_BY_ID = new Map(SEATS.map((seat) => [seat.id, seat]));

export function seatName(id: SeatId) {
  return SEAT_BY_ID.get(id)?.name ?? "STUDENT";
}
