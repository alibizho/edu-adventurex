import { useMemo } from "react";
import { SEATS, type SeatId } from "./classroom.seats";

/**
 * Who is actually sitting in the room this session.
 *
 * Clicking a `?` used to zoom in on the same generic body no matter whose hand it was, so the six
 * students were six names attached to one person. Each seat now draws its own sprite, dealt once
 * and held for the rest of the session — MILA is the same student every time you answer her, and
 * still is after a reload.
 *
 * The room itself is unchanged: the marker over a seat stays a plain `?`. This is only about who
 * is waiting when the camera arrives.
 *
 * Dealt randomly rather than hardcoded so the class isn't the same six people every time.
 */
const SPRITES: readonly string[] = [1, 2, 3, 4, 5].map((n) => `/images/students/${n}.png`);

export type ClassroomCast = Readonly<Record<SeatId, string>>;

function shuffled<T>(items: readonly T[]): T[] {
  const deck = [...items];
  for (let i = deck.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [deck[i], deck[j]] = [deck[j], deck[i]];
  }
  return deck;
}

/** The seats close enough that reusing a sprite across them reads as a bug, not a coincidence. */
function neighbours(seatId: SeatId): SeatId[] {
  const columns = ["left", "mid", "right"];
  const [row, column] = seatId.split("-");
  const index = columns.indexOf(column);
  return [
    `${row === "back" ? "front" : "back"}-${column}`,
    ...(index > 0 ? [`${row}-${columns[index - 1]}`] : []),
    ...(index < columns.length - 1 ? [`${row}-${columns[index + 1]}`] : []),
  ] as SeatId[];
}

/**
 * One sprite per seat, as distinct as the pool allows.
 *
 * Dealing from a shuffled deck rather than choosing per seat is what stops the room seating three
 * copies of the same student. There are currently five sprites for six seats, so one of them does
 * get a twin — add a sixth file and this loop simply stops needing the second branch — and the
 * twin is seated away from the original: side by side or one behind the other, a repeat looks
 * like the same sprite drawn twice rather than two people who dress alike.
 */
function deal(): ClassroomCast {
  const deck = shuffled(SPRITES);
  const cast: Partial<Record<SeatId, string>> = {};
  SEATS.forEach((seat, index) => {
    if (index < deck.length) {
      cast[seat.id] = deck[index];
      return;
    }
    const nearby = new Set(neighbours(seat.id).map((id) => cast[id]));
    cast[seat.id] = shuffled(SPRITES).find((sprite) => !nearby.has(sprite)) ?? deck[index % deck.length];
  });
  return cast as ClassroomCast;
}

function isCast(value: unknown): value is ClassroomCast {
  if (!value || typeof value !== "object") return false;
  const record = value as Record<string, unknown>;
  return SEATS.every((seat) => SPRITES.includes(record[seat.id] as string));
}

/**
 * The cast for one class, stable for as long as the tab is open.
 *
 * Held in sessionStorage rather than component state because the room is unmounted every time the
 * learner zooms into a student, and re-dealing on the way back would give the class a new set of
 * faces mid-lesson — including for the student they are in the middle of answering. The stored
 * cast is validated on read: one written against a different sprite set is simply re-dealt.
 */
export function useClassroomCast(classKey: string): ClassroomCast {
  return useMemo(() => {
    const storageKey = `wut:cast:${classKey}`;
    try {
      const saved: unknown = JSON.parse(sessionStorage.getItem(storageKey) ?? "null");
      if (isCast(saved)) return saved;
    } catch {
      // Storage is unavailable or holds junk. A fresh cast is a fine answer to both.
    }
    const cast = deal();
    try {
      sessionStorage.setItem(storageKey, JSON.stringify(cast));
    } catch {
      // Not worth failing the room over: the cast just won't survive a reload.
    }
    return cast;
  }, [classKey]);
}
