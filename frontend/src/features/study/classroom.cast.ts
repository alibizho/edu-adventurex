import { useMemo } from "react";
import { SEATS, type SeatId } from "./classroom.seats";

/**
 * Who is actually sitting in the room this session, and how they are sitting.
 *
 * Two things, dealt together because both must survive the whole class. The **sprite** is who you
 * meet when you press a `?` — the room used to zoom in on the same generic body no matter whose
 * hand it was, so six names shared one person. The **resting pose** is what the seat looks like
 * from the teacher's desk: mostly heads-up, one or two asleep, which is what makes the room read
 * as a class rather than a row of identical figures.
 *
 * Both are dealt at random rather than hardcoded, so no two classes are the same room, and both
 * are held for the session: a student who was asleep when you started is not sitting up straight
 * the next time you glance over.
 */
const SPRITES: readonly string[] = [1, 2, 3, 4, 5].map((n) => `/images/students/${n}.png`);

/** How a seat sits when nothing is happening. A raised hand overrides this — see POSE_SOURCE. */
export type RestingPose = "idle" | "sleeping";

/** Every pose a seat can be drawn in, including the one no one rests in. */
export type SeatPose = RestingPose | "handup";

export const POSE_SOURCE: Readonly<Record<SeatPose, string>> = {
  idle: "/images/idle.png",
  sleeping: "/images/sleeping.png",
  handup: "/images/handup.png",
};

export type SeatCast = { sprite: string; resting: RestingPose };
export type ClassroomCast = Readonly<Record<SeatId, SeatCast>>;

/**
 * Exactly two of the six are asleep — which two is what changes between classes.
 *
 * A fixed count rather than a random one: the room should read the same way every time you walk
 * into it (four heads up, two down), and rolling the number as well as the seats meant some
 * classes opened with a single sleeper and looked like a different room.
 */
const ASLEEP = 2;

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
 * One sprite per seat, as distinct as the pool allows, plus who is asleep.
 *
 * Dealing sprites from a shuffled deck rather than choosing per seat is what stops the room seating
 * three copies of the same student. There are currently five sprites for six seats, so one of them
 * does get a twin — add a sixth file and this loop stops needing the second branch — and the twin
 * is seated away from the original: side by side or one behind the other, a repeat looks like the
 * same sprite drawn twice rather than two people who dress alike.
 */
function deal(): ClassroomCast {
  const deck = shuffled(SPRITES);
  const sprites: Partial<Record<SeatId, string>> = {};
  SEATS.forEach((seat, index) => {
    if (index < deck.length) {
      sprites[seat.id] = deck[index];
      return;
    }
    const nearby = new Set(neighbours(seat.id).map((id) => sprites[id]));
    sprites[seat.id] = shuffled(SPRITES).find((sprite) => !nearby.has(sprite)) ?? deck[index % deck.length];
  });

  const asleep = new Set(shuffled(SEATS.map((seat) => seat.id)).slice(0, ASLEEP));

  return Object.fromEntries(SEATS.map((seat) => [seat.id, {
    sprite: sprites[seat.id]!,
    resting: asleep.has(seat.id) ? "sleeping" : "idle",
  }])) as ClassroomCast;
}

function isCast(value: unknown): value is ClassroomCast {
  if (!value || typeof value !== "object") return false;
  const record = value as Record<string, Partial<SeatCast> | undefined>;
  const wellFormed = SEATS.every((seat) => {
    const entry = record[seat.id];
    return Boolean(entry)
      && SPRITES.includes(entry!.sprite as string)
      && (entry!.resting === "idle" || entry!.resting === "sleeping");
  });
  // The sleeper count is checked, not just the shape: a cast dealt under a different rule is still
  // well-formed, and without this a tab open from before the rule changed keeps its old room.
  return wellFormed
    && SEATS.filter((seat) => record[seat.id]!.resting === "sleeping").length === ASLEEP;
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
