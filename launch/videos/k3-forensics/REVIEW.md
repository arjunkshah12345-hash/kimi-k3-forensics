# Review: K3 Weight Forensics launch film (40.2 s, 1920x1080, 30 fps)

## Creative calls
- Built storyline A ("Too big to run, so we read it"): size vs GPU, the byte-range trick, title, the lattice printing from real per-layer data, a zoom into layer 84 that becomes the expert card, the 16 untrained tokens, end card.
- Not built: B (result-first pull-back from one expert) and C (kinetic type on the beat, needs music). Both in STORYLINES.md.
- Look reuses the site: #121614 ground, Geist / Geist Mono, ember #F2A65A rationed to the GPU sliver, the requested shard, lit lattice cells, finding numbers and the mark.

## What was measured
- `hyperframes check`: ok, no warnings after fixes (line holds lengthened, riser moved to its audible part, exit tweens made non-immediate).
- Review sheets checked beat by beat. Fixed a P0 where exit `fromTo`s rendered immediately and overwrote entrance states (labels showed early, payoff cards were blank), and a P0 where a leading-space marker rendered like an underscore (` java` vs `_java`): leading space is now a muted `·`.
- Rendered audio: true peak −1 dBTP, no clipping, −12.1 LUFS integrated; the payoff impact peaks at 23.57 s against the zoom at 23.5 s. Measured, not heard.

## Sound
No music: no HeyGen sign-in for catalog music. Bundled Pixabay-licensed effects only (whooshes on seams, typing on the request, riser into one bass impact at the payoff, pops on landings). To add a bed, drop an `<audio id="bgm">` at volume ~0.35 into index.html.

## Weakest shot
The first ~0.4 s of the lattice scene is nearly empty while the first columns print. With more time: start the lattice at ~15% printed, or carry the title's facts line into it.
