import { useId } from "react"

import { findings } from "@/data/findings"
import { FillPattern } from "@/lib/dither"

const VOCAB = 163_584
const W = 1000
const H = 56

/**
 * The whole BPE vocabulary as one strip, left to right by token id, with the 16
 * never-trained tokens marked. Most sit at the very start (raw bytes and id 1699), two at
 * ids 57,118 and 57,119.
 */
export function VocabStrip() {
  const id = useId().replace(/:/g, "")
  const marks = findings.untrained.tokens
  return (
    <figure className="flex flex-col gap-2">
      <svg
        viewBox={`0 0 ${W} ${H + 18}`}
        className="h-auto w-full"
        role="img"
        aria-label={`The ${VOCAB.toLocaleString()} token vocabulary with the ${marks.length} untrained tokens marked: 13 raw bytes at ids 124 to 187, one at id 1699, two adjacent tokens at ids 57118 and 57119.`}
      >
        <defs>
          <FillPattern id={`${id}-trained`} fill="dotted" color="ink" cell={3} />
        </defs>
        <rect x={0} y={8} width={W} height={H - 16} fill={`url(#${id}-trained)`} opacity={0.55} />
        {marks.map((m) => {
          const x = (m.id / VOCAB) * W
          return <rect key={m.id} x={Math.max(0, x - 1.5)} y={0} width={3} height={H} fill="var(--dither-color-orange)" />
        })}
        <text x={0} y={H + 15} className="fill-muted-foreground font-mono text-[11px]">id 0</text>
        <text x={(57118 / VOCAB) * W} y={H + 15} textAnchor="middle" className="fill-accent font-mono text-[11px]">57,118</text>
        <text x={W} y={H + 15} textAnchor="end" className="fill-muted-foreground font-mono text-[11px]">163,583</text>
      </svg>
      <figcaption className="text-sm text-muted-foreground">
        Every BPE token, by id. Orange marks the 16 that never received a gradient on the input side.
      </figcaption>
    </figure>
  )
}
