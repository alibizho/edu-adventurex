"""POC fixtures — one topic (sliding friction), taught two ways.

The SOURCE is ground truth. Answer keys derive from it, NOT from the transcripts. GOOD teaches
it accurately; BAD teaches confident misconceptions. A working transfer delta should come back
positive-ish on GOOD and zero-or-negative on BAD (teaching made the learner worse).

Transcripts are lists of child utterances; the harness turns them into numbered Segments.
Gold questions are transfer questions with keys grounded in SOURCE.
"""

TOPIC = "Sliding friction — how far a box slides before stopping"

# --- Ground truth. Answer keys are grounded here. ---
SOURCE = """\
Friction is a force between two surfaces in contact that opposes their relative motion. When a
box slides across a floor, kinetic friction points opposite to the direction of sliding and
slows the box down.

The size of the friction force is f = mu * N, where N is the normal force (on flat ground, N
equals the object's weight) and mu is the friction coefficient, which depends on the pair of
surfaces. Rougher surface pairs have a larger mu (more friction); smoother pairs like ice on
metal have a much smaller mu (less friction).

A key and counterintuitive result: on flat ground the deceleration of a sliding object is
a = f / m = (mu * m * g) / m = mu * g, which does NOT depend on the object's mass. A heavier box
has more friction force, but it also has more inertia, and the two cancel. So two boxes of
different weight, started at the same speed on the same surface, decelerate at the same rate and
slide roughly the same distance before stopping.

The stopping distance is d = v^2 / (2 * mu * g). It grows with the square of the initial speed
(twice the speed means four times the distance) and shrinks when mu is larger (rougher surface,
or more friction, stops the box sooner). Kinetic energy lost to friction is converted mostly to
heat; friction does negative work on the box and removes its energy — it never adds energy.
"""

# --- GOOD: accurate explanation of the source. ---
GOOD_TRANSCRIPT = [
    "Okay so friction is a force between two surfaces that are touching, and it fights the motion.",
    "When a box slides to the right, friction pushes to the left, opposite to the way it's sliding, so it slows down.",
    "How much friction you get depends on the two surfaces — rough surfaces give more friction, smooth ones like ice give a lot less.",
    "Here's the surprising part: on flat ground, how fast the box slows down doesn't depend on how heavy it is.",
    "A heavy box has more friction, but it's also harder to slow down because it has more inertia, and those two exactly cancel out.",
    "So two boxes at the same starting speed on the same floor, one light and one heavy, slide about the same distance before stopping.",
    "If you slide it faster, it goes much farther — double the speed is four times the distance, because distance depends on speed squared.",
    "And if the floor is rougher, it stops sooner; if you put it on ice, there's way less friction so it slides much farther.",
    "The energy doesn't vanish — friction turns the box's motion energy into heat as it drags to a stop.",
]

# --- BAD: confident misconceptions that contradict the source. ---
BAD_TRANSCRIPT = [
    "So friction is basically about weight — the heavier something is, the more it grips the floor.",
    "That means a heavier box always stops way sooner than a light one, because it has so much more friction dragging it down.",
    "Also smoother surfaces actually have more friction, because a smooth flat surface grips better and touches more of the box.",
    "So a box slides shorter on smooth polished wood than on rough carpet, since the smooth floor holds it tighter.",
    "Friction also gives the box a little push in the direction it's already going, that's part of why it keeps sliding for a while.",
    "And when you slide something faster, it stops in a shorter distance, because friction works harder the faster you go.",
    "On ice you stop really quickly, because cold ice is sticky and grabs the box.",
    "When the box stops, the energy just disappears — it's gone, that's why it stops.",
]

# --- Gold transfer questions. Keys grounded in SOURCE (ground truth), never the transcripts. ---
GOLD_QUESTIONS = [
    {
        "text": "Two boxes start sliding across the same flat floor at the same speed. Box A weighs 2 kg, Box B weighs 8 kg. Which slides a longer distance before stopping, or about the same? Explain.",
        "answer_key": "About the same distance. On flat ground the friction deceleration is mu*g, which does not depend on mass, so both boxes stop after roughly the same distance.",
    },
    {
        "text": "A kid claims 'heavier things always stop sooner when sliding.' On a flat floor, same surface, same starting speed, is that right? Why or why not?",
        "answer_key": "No. A heavier object has more friction force but also more inertia; the two cancel, so the stopping distance is about the same regardless of weight on flat ground.",
    },
    {
        "text": "You slide a box across rough carpet, then across smooth polished wood at the same speed. On which does it travel farther, and why?",
        "answer_key": "Farther on the smooth polished wood, because smoother surfaces have a smaller friction coefficient, so less force opposes the motion.",
    },
    {
        "text": "While a box slides to the right and slows down, which direction does the friction force point?",
        "answer_key": "To the left — kinetic friction opposes the relative motion, so it points opposite to the direction of sliding.",
    },
    {
        "text": "If you doubled the roughness (friction coefficient) of the floor, what happens to how far a sliding box travels at the same starting speed?",
        "answer_key": "It travels a shorter distance — a larger friction coefficient means a larger deceleration, so the box stops sooner.",
    },
    {
        "text": "A box slides across a floor and stops. Someone says friction 'gave energy' to the box. Is that right, and where did the box's kinetic energy go?",
        "answer_key": "No. Friction removed the box's kinetic energy and converted it mostly to heat; friction does negative work on the box, it never adds energy.",
    },
    {
        "text": "On the same flat surface, does a box sliding at a higher initial speed stop in a shorter or longer distance than one at a lower speed?",
        "answer_key": "Longer distance — stopping distance is d = v^2/(2*mu*g), so it grows with the square of the initial speed; faster boxes slide farther before stopping.",
    },
    {
        "text": "You move the same box from concrete onto ice at the same starting speed. Does it slide farther or less far, and why?",
        "answer_key": "Farther, because ice has a much smaller friction coefficient than concrete, so there is less deceleration.",
    },
]
