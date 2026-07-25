export type SeatId =
  | "back-left" | "back-mid" | "back-right"
  | "front-left" | "front-mid" | "front-right";

export type ClassroomSeat = {
  id: SeatId;
  name: string;
  x: number;
  base: number;
};

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
