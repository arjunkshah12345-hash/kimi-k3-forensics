import { type ReactNode, type RefObject, useEffect, useRef, useState } from "react"

import { findings } from "@/data/findings"
import { DitherShader } from "@/lib/dither"

// The hero is the main finding drawn as a lattice: one column per MoE layer
// (left = layer 1, right = layer 92). Each column's dot density is that layer's
// measured share of "spelling" experts, so the right edge fills in exactly where
// the data does. A slow scan line sweeps across, like a range request reading
// the checkpoint shard by shard.
const SHARES = findings.writers.per_layer.map((p) => p.share)
const N = SHARES.length
const PACKED: [number, number, number, number][] = []
for (let i = 0; i < N; i += 4) {
  PACKED.push([SHARES[i] ?? 0, SHARES[i + 1] ?? 0, SHARES[i + 2] ?? 0, SHARES[i + 3] ?? 0])
}
const UNIFORMS = Object.fromEntries(PACKED.map((v, i) => [`share${i}`, v]))
const LOOKUP = PACKED.map(
  (_, i) => `  if (q < ${i}.5) { v = share${i}; }`
).join(" else\n")

const FIELD = `
float latticeHash(vec2 p) {
  p = fract(p * vec2(234.34, 435.345));
  p += dot(p, p + 34.23);
  return fract(p.x * p.y);
}

float layerShare(float i) {
  float q = floor(i / 4.0);
  vec4 v = vec4(0.0);
${LOOKUP}
  float r = mod(i, 4.0);
  return r < 0.5 ? v.x : r < 1.5 ? v.y : r < 2.5 ? v.z : v.w;
}

float field(vec2 uv, vec2 res, float t) {
  float cols = ${N}.0;
  float col = floor(uv.x * cols);
  float colW = res.x / cols;
  float rows = floor(res.y / max(colW, 1.0));
  float row = floor(uv.y * rows);
  vec2 inCell = fract(vec2(uv.x * cols, uv.y * rows));

  // A square dot centred in each lattice cell.
  float dotMask = step(0.28, inCell.x) * step(inCell.x, 0.72) * step(0.28, inCell.y) * step(inCell.y, 0.72);

  float share = layerShare(col);
  float lit = step(latticeHash(vec2(col, row)), 0.015 + share * 1.9);

  // Slow scan line moving left to right.
  float scan = fract(t * 0.035);
  float dx = uv.x - scan;
  float beam = exp(-dx * dx * 900.0);

  float swept = step(0.5, beam) * step(latticeHash(vec2(row, col)), 0.35);
  float tone = dotMask * max(lit, swept);

  vec2 d = abs((uv - heroBox.xy) * res) - heroBox.zw * res + 10.0;
  float dist = length(max(d, 0.0)) + min(max(d.x, d.y), 0.0) - 10.0;
  float clear = smoothstep(0.0, 22.0, dist);
  return clamp(tone, 0.0, 1.0) * clear * smoothstep(0.0, 0.18, uv.y);
}
`

type Box = readonly [number, number, number, number]

function useClearBox(section: RefObject<HTMLElement | null>, text: RefObject<HTMLElement | null>, pad = 28): Box {
  const [box, setBox] = useState<Box>([0.5, 0.5, 0, 0])
  useEffect(() => {
    const host = section.current
    const block = text.current
    if (!(host && block)) return
    const measure = () => {
      const s = host.getBoundingClientRect()
      const r = block.getBoundingClientRect()
      if (s.width === 0 || s.height === 0) return
      setBox([
        (r.left - s.left + r.width / 2) / s.width,
        1 - (r.top - s.top + r.height / 2) / s.height,
        (r.width / 2 + pad) / s.width,
        (r.height / 2 + pad) / s.height,
      ])
    }
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(host)
    observer.observe(block)
    return () => observer.disconnect()
  }, [section, text, pad])
  return box
}

export function Hero({ children }: { children: ReactNode }) {
  const section = useRef<HTMLElement>(null)
  const text = useRef<HTMLDivElement>(null)
  const box = useClearBox(section, text)
  return (
    <section ref={section} className="relative isolate overflow-hidden">
      <DitherShader
        className="absolute inset-0 -z-10"
        field={FIELD}
        palette={["--dither-paper", "orange"]}
        cell={3}
        stillTime={9}
        uniforms={{ heroBox: box, ...UNIFORMS }}
        fallback={<div className="absolute inset-0 bg-gradient-to-l from-dither-orange/20 to-transparent" />}
      />
      <div className="mx-auto flex min-h-[620px] max-w-6xl items-center px-[max(16px,env(safe-area-inset-left,0px))] py-24 sm:min-h-[700px]">
        <div ref={text} className="w-fit max-w-2xl">
          {children}
        </div>
      </div>
      <div className="pointer-events-none absolute inset-x-0 bottom-3 mx-auto flex max-w-6xl justify-between px-[max(16px,env(safe-area-inset-left,0px))] font-mono text-[11px] text-muted-foreground">
        <span>layer 1</span>
        <span className="hidden sm:inline">each column is one MoE layer · density = share of spelling experts</span>
        <span>layer 92</span>
      </div>
    </section>
  )
}
