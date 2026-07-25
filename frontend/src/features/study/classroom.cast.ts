import { useMemo } from "react";
import { SEATS, type SeatId } from "./classroom.seats";

const SPRITES: readonly string[] = [1, 2, 3, 4, 5].map((n) => `/images/students/${n}.png`);

export type RestingPose = "idle" | "sleeping";

export type SeatPose = RestingPose | "handup";

export const POSE_SOURCE: Readonly<Record<SeatPose, string>> = {
  idle: "/images/idle.png",
  sleeping: "/images/sleeping.png",
  handup: "/images/handup.png",
};

export type SeatCast = { sprite: string; resting: RestingPose };
export type ClassroomCast = Readonly<Record<SeatId, SeatCast>>;

const ASLEEP = 2;

function shuffled<T>(items: readonly T[]): T[] {
  const deck = [...items];
  for (let i = deck.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [deck[i], deck[j]] = [deck[j], deck[i]];
  }
  return deck;
}

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
  return wellFormed
    && SEATS.filter((seat) => record[seat.id]!.resting === "sleeping").length === ASLEEP;
}

export function useClassroomCast(classKey: string): ClassroomCast {
  return useMemo(() => {
    const storageKey = `wut:cast:${classKey}`;
    try {
      const saved: unknown = JSON.parse(sessionStorage.getItem(storageKey) ?? "null");
      if (isCast(saved)) return saved;
    } catch {
    }
    const cast = deal();
    try {
      sessionStorage.setItem(storageKey, JSON.stringify(cast));
    } catch {
    }
    return cast;
  }, [classKey]);
}
