import { AreaChart } from "@/components/dither-kit/area-chart"
import { Area } from "@/components/dither-kit/area"
import { BarChart } from "@/components/dither-kit/bar-chart"
import { Bar } from "@/components/dither-kit/bar"
import { Grid } from "@/components/dither-kit/grid"
import { Tooltip } from "@/components/dither-kit/tooltip"
import { XAxis } from "@/components/dither-kit/x-axis"
import { YAxis } from "@/components/dither-kit/y-axis"
import { findings } from "@/data/findings"

const pct = (v: number) => `${Math.round(v * 100)}%`

const spelling = findings.writers.per_layer.map((p) => ({ layer: `L${p.layer}`, share: p.share }))

export function SpellingChart() {
  return (
    <div
      className="h-[300px] w-full sm:h-[340px]"
      role="img"
      aria-label={`Share of spelling experts per layer: about ${pct(findings.writers.early_share)} in layers 1 to 47, ${pct(findings.writers.late_share)} in layers 82 to 91, peaking at ${pct(findings.writers.peak.share)} in layer ${findings.writers.peak.layer}.`}
    >
      <BarChart data={spelling} config={{ share: { label: "Spelling experts", color: "orange" } }} bloom="low">
        <Grid />
        <XAxis dataKey="layer" maxTicks={8} />
        <YAxis tickFormatter={pct} tickCount={5} />
        <Tooltip labelKey="layer" valueFormatter={(v) => `${(v * 100).toFixed(1)}% of 896 experts`} />
        <Bar dataKey="share" variant="gradient" />
      </BarChart>
    </div>
  )
}

const memory = findings.kda.per_layer.map((p) => ({ layer: `L${p.layer}`, halfLife: p.median_half_life }))

export function MemoryChart() {
  return (
    <div className="h-[260px] w-full" role="img" aria-label="Median resting half-life per Kimi Delta Attention layer, rising from 4 tokens at layer 0 to 117 tokens at layer 86.">
      <AreaChart data={memory} config={{ halfLife: { label: "Median half-life", color: "blue" } }} bloom="low">
        <Grid />
        <XAxis dataKey="layer" maxTicks={8} />
        <YAxis tickFormatter={(v) => `${v}`} tickCount={4} />
        <Tooltip labelKey="layer" valueFormatter={(v) => `${v.toFixed(1)} tokens`} />
        <Area dataKey="halfLife" variant="gradient" />
      </AreaChart>
    </div>
  )
}

const suppression = findings.bias.per_layer.map((p) => ({ layer: `L${p.layer}`, push: -p.min, boost: p.max }))

export function BiasChart() {
  return (
    <div className="h-[260px] w-full" role="img" aria-label="Strongest suppression and strongest boost from the balancing bias per MoE layer. Suppression grows to 0.76 at layer 90; boosts stay under 0.15.">
      <BarChart
        data={suppression}
        stackType="stacked"
        config={{
          push: { label: "Strongest push down", color: "red" },
          boost: { label: "Strongest boost", color: "green" },
        }}
        bloom="low"
      >
        <Grid />
        <XAxis dataKey="layer" maxTicks={8} />
        <YAxis tickFormatter={(v) => v.toFixed(1)} tickCount={4} />
        <Tooltip labelKey="layer" valueFormatter={(v, name) => `${name === "push" ? "−" : "+"}${v.toFixed(3)}`} />
        <Bar dataKey="push" variant="gradient" />
        <Bar dataKey="boost" variant="solid" />
      </BarChart>
    </div>
  )
}
