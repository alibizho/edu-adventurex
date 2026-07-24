# Teachable Student — What This Project Is

## In one line

A kid learns a topic, then **teaches it to an AI that knows nothing** — and we *measure*
how well they actually understood it, instead of just grading them.

## The problem

Most AI education tools ask a kid questions and a model gives them a score. But a smooth
answer can *feel* like understanding when nothing was really learned, and that score is just
one model's opinion. We wanted a real measurement, not an opinion.

## The idea

You understand something best when you have to teach it. So we flip the roles:

- The **AI is the student**. It knows nothing about the topic.
- The **kid is the teacher**. They explain it out loud, using a source they studied.
- The AI acts like a confused classmate: it only **asks questions, repeats things back, or
  says it's lost**. It never explains and never gives away the answer.

While the kid teaches, we measure their understanding through three separate channels. Each
one catches a different kind of failure, so together they're hard to fool.

## The three measurements

**1. Did the explanation actually land? (the main one)**
After the lesson, the AI takes a test the kid never sees. Two groups take it: one that
*heard the kid's lesson*, and one that *heard nothing*.
- Heard-the-lesson group does better → the teaching worked.
- Both do the same → the teaching added nothing.
- Heard-the-lesson group does **worse** → the kid taught something wrong and actually made
  the learner worse. (No normal grading tool can show you this.)

**2. Did the kid *sound* sure? (from their voice)**
While they talk, we listen for signs of uncertainty — pauses, "um"s, "kind of" / "I think",
shaky or rising pitch, self-corrections.

**3. Does what they say hold together? (checked live)**
As the kid speaks, we check each sentence against what they said earlier and against the
source. If they contradict themselves or the source, we catch it in the moment.

## The payoff

We cross **"how sure they sounded"** with **"did it actually work"**. The interesting case:

> The kid sounded totally confident — and was wrong.

That's a **blind spot** — the exact thing a kid doesn't know they don't know, and the thing
no rubric can see. Finding those is the whole point.

## Quick example

A kid teaches "a heavy box slides less far than a light one." They say, a little shakily,
*"the box stops because it's heavy."* The source actually says both boxes slow down at the
same rate.

- Their **voice** sounded unsure right there.
- The **live check** notices it contradicts the source.
- The **test afterward** confirms the learners who heard this lesson did *worse* than those
  who heard nothing.

All three point at the same sentence. On screen, that sentence lights up red next to the
correct source text — and the kid can try teaching it again.

## Why it's more than "an AI wrapper"

The answer never comes from asking a model "how good was this?" It comes from an
**experiment** — a group that heard the lesson vs. a group that didn't — plus a voice model
and a live fact-check. The AI is the *student being tested*, not the judge.