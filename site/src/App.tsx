import type { ReactNode } from "react"

import { BiasChart, MemoryChart, SpellingChart } from "@/components/site/charts"
import { Hero } from "@/components/site/hero"
import { VocabStrip } from "@/components/site/vocab-strip"
import { findings, MODEL_URL, REPO_URL, showToken } from "@/data/findings"

const pct = (v: number) => `${Math.round(v * 100)}%`
const gutter = "px-[max(16px,env(safe-area-inset-left,0px))]"

function Section({ id, children, wide = false }: { id?: string; children: ReactNode; wide?: boolean }) {
  return (
    <section id={id} className={`mx-auto w-full ${wide ? "max-w-6xl" : "max-w-4xl"} ${gutter} scroll-mt-16 py-16 sm:py-24`}>
      {children}
    </section>
  )
}

function Prose({ children }: { children: ReactNode }) {
  return <div className="flex max-w-[65ch] flex-col gap-4 text-[17px] leading-relaxed text-foreground/85">{children}</div>
}

function Tok({ t, tone = "promote" }: { t: string; tone?: "promote" | "opposite" }) {
  return (
    <span
      className={`whitespace-pre rounded-sm px-1.5 py-0.5 font-mono text-[12px] ${
        tone === "promote" ? "bg-dither-orange/15 text-dither-orange" : "bg-dither-blue/15 text-dither-blue"
      }`}
    >
      {showToken(t)}
    </span>
  )
}

function Code({ children }: { children: ReactNode }) {
  return <code className="rounded-sm bg-muted px-1.5 py-0.5 font-mono text-[0.88em]">{children}</code>
}

const untrainedNote = (t: string) =>
  /^\\x(c[01]|f[5-9a-f])$/.test(t)
    ? "byte that can never appear in valid UTF-8"
    : t.includes("Bmori")
      ? "looks like a species code (Bombyx mori)"
      : t.includes("Pnapi")
        ? "looks like a species code (Pieris napi)"
        : "a broken Cyrillic byte pair"

export default function App() {
  const w = findings.writers
  const sim = findings.experts.similarity
  return (
    <div className="min-h-screen">
      <header className="sticky top-[env(safe-area-inset-top,0px)] z-20 border-b border-border/60 bg-background/85 backdrop-blur">
        <nav className={`mx-auto flex h-14 max-w-6xl items-center justify-between ${gutter}`}>
          <a href="#top" className="font-medium tracking-tight">K3 Weight Forensics</a>
          <div className="flex items-center gap-5 text-sm text-muted-foreground">
            <a className="hover:text-foreground focus-visible:text-foreground" href="#findings">Findings</a>
            <a className="hidden hover:text-foreground focus-visible:text-foreground sm:inline" href="#method">Method</a>
            <a className="rounded-md border border-border px-3 py-1.5 text-foreground hover:border-foreground/40" href={REPO_URL}>GitHub</a>
          </div>
        </nav>
      </header>

      <main id="top">
        <Hero>
          <div className="flex flex-col gap-6">
            <h1 className="text-[clamp(2.4rem,6.4vw,4.4rem)] font-semibold leading-[1.02] tracking-[-0.03em]">
              Reading Kimi K3 without running it
            </h1>
            <p className="max-w-xl text-lg leading-relaxed text-foreground/80">
              K3 has 2.8 trillion parameters, far too many for a free GPU. We pulled {findings.run.downloaded_gb} GB of
              tensors out of its checkpoint with HTTP range requests and studied the weights on one Kaggle T4. Late in the
              network, up to half the experts turn out to be spelling experts.
            </p>
            <div className="flex flex-wrap gap-3">
              <a href="#findings" className="rounded-md bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90">
                Read the findings
              </a>
              <a href={REPO_URL} className="rounded-md border border-border bg-background/60 px-4 py-2.5 text-sm font-medium hover:border-foreground/40">
                Get the code and data
              </a>
            </div>
          </div>
        </Hero>

        <div id="findings" />
        <Section wide>
          <div className="flex flex-col gap-10">
            <div className="flex flex-col gap-5">
              <h2 className="text-3xl font-semibold tracking-tight sm:text-4xl">Late experts become spelling experts</h2>
              <Prose>
                <p>
                  For each of the {w.experts_scanned.toLocaleString()} routed experts, we found the direction it writes most
                  strongly and read that direction out through the unembedding. Then we checked one thing: do at least 5 of
                  its top 10 tokens belong to one word family, like <Code> java</Code>, <Code>-java</Code>, <Code>_java</Code>, <Code>Java</Code>?
                </p>
                <p>
                  In layers 1 to 47, about {pct(w.early_share)} of experts do. In layers 82 to 91 it is {pct(w.late_share)}, and
                  layer {w.peak.layer} reaches {pct(w.peak.share)}. Each of these experts picks the surface form of one word: its
                  casing, its leading space, the punctuation glued to it.
                </p>
              </Prose>
            </div>
            <SpellingChart />
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {w.gallery.slice(0, 12).map((g) => (
                <article key={`${g.layer}-${g.expert}`} className="flex min-w-0 flex-col gap-3 rounded-lg border border-border bg-card p-4">
                  <span className="font-mono text-xs text-muted-foreground">layer {g.layer} · expert {g.expert}</span>
                  <div className="flex flex-wrap gap-1.5">{g.promotes.map((t, i) => <Tok key={i} t={t} />)}</div>
                  <div className="flex flex-wrap gap-1.5 border-t border-border/60 pt-3">
                    {g.opposite.map((t, i) => <Tok key={i} t={t} tone="opposite" />)}
                  </div>
                </article>
              ))}
            </div>
            <p className="text-sm text-muted-foreground">
              Orange: tokens the expert's main write direction promotes. Blue: the opposite end of the same direction, which is
              often a second word family.
            </p>
          </div>
        </Section>

        <Section>
          <div className="flex flex-col gap-8">
            <h2 className="text-3xl font-semibold tracking-tight sm:text-4xl">Only 16 tokens were never trained</h2>
            <Prose>
              <p>
                The usual glitch-token test compares every token with the reserved ones, assuming untrained rows drift
                together. In K3 they don't. The 240 reserved rows are still untouched random initialization: kurtosis{" "}
                {findings.untrained.stats.embed.reserved_kurtosis_mean.toFixed(2)}, a perfect Gaussian, and nearest-neighbour
                cosine around {findings.untrained.stats.embed.reserved_nncos_p50.toFixed(2)}.
              </p>
              <p>
                A trained token has close neighbours (median {findings.untrained.stats.embed.bpe_nncos_p50.toFixed(2)}). Testing
                for that instead flags 16 of 163,584 input tokens. Thirteen are bytes that valid UTF-8 never contains and one is a
                broken Cyrillic byte pair. The last two are an adjacent pair, <Code>⇥Bmori</Code> and <Code>⇥Pnapi</Code>, which read like genome-annotation species
                codes that reached the tokenizer's corpus but not the training data. On the output side, nothing is untrained.
              </p>
            </Prose>
            <VocabStrip />
            <div className="overflow-x-auto rounded-lg border border-border">
              <table className="w-full min-w-[520px] text-sm">
                <thead className="bg-card text-left text-muted-foreground">
                  <tr>
                    <th className="px-4 py-2.5 font-medium">Token id</th>
                    <th className="px-4 py-2.5 font-medium">Token</th>
                    <th className="px-4 py-2.5 font-medium">Nearest-neighbour cosine</th>
                    <th className="px-4 py-2.5 font-medium">What it is</th>
                  </tr>
                </thead>
                <tbody className="tabular-nums">
                  {findings.untrained.tokens.map((u) => (
                    <tr key={u.id} className="border-t border-border/60">
                      <td className="px-4 py-2 font-mono">{u.id}</td>
                      <td className="px-4 py-2 font-mono">{showToken(u.token)}</td>
                      <td className="px-4 py-2 font-mono">{u.nn_cos.toFixed(3)}</td>
                      <td className="px-4 py-2 text-muted-foreground">{untrainedNote(u.token)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </Section>

        <Section wide>
          <div className="grid grid-cols-1 items-center gap-10 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
            <div className="flex min-w-0 flex-col gap-5">
              <h2 className="text-3xl font-semibold tracking-tight sm:text-4xl">Memory gets longer with depth</h2>
              <Prose>
                <p>
                  69 of K3's 93 layers use Kimi Delta Attention, a linear attention whose memory decays channel by channel.
                  The weights <Code>A_log</Code> and <Code>dt_bias</Code> fix each channel's resting decay, which converts to
                  a half-life in tokens.
                </p>
                <p>
                  The median half-life climbs from 4 tokens in layer 0 to 117 in layer 86. Across all layers{" "}
                  {pct(findings.kda.share_over_1k)} of channels hold on for more than 1,000 tokens.
                </p>
              </Prose>
            </div>
            <div className="min-w-0 rounded-lg border border-border bg-card p-4">
              <MemoryChart />
            </div>
          </div>
        </Section>

        <Section>
          <div className="flex flex-col gap-8">
            <h2 className="text-3xl font-semibold tracking-tight sm:text-4xl">No two experts are alike</h2>
            <Prose>
              <p>
                We decoded every expert in layers 1, 46 and 91 from MXFP4 and ran all 2,688 of them on the same 4,096 real
                token inputs, then compared every pair by output. No pair comes close to a duplicate, so there is nothing to
                prune by merging twins.
              </p>
            </Prose>
            <dl className="grid grid-cols-1 gap-6 sm:grid-cols-3">
              {(["1", "46", "91"] as const).map((k) => (
                <div key={k} className="flex flex-col gap-1 border-t border-border pt-4">
                  <dt className="text-sm text-muted-foreground">Layer {k}, most similar pair</dt>
                  <dd className="text-4xl font-semibold tabular-nums tracking-tight text-accent">{sim[k].max_pair.toFixed(2)}</dd>
                  <dd className="text-sm text-muted-foreground">typical nearest neighbour {sim[k].mean_nearest.toFixed(3)}</dd>
                </div>
              ))}
            </dl>
          </div>
        </Section>

        <Section wide>
          <div className="grid grid-cols-1 items-center gap-10 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
            <div className="order-2 min-w-0 rounded-lg border border-border bg-card p-4 lg:order-1">
              <BiasChart />
            </div>
            <div className="order-1 flex min-w-0 flex-col gap-5 lg:order-2">
              <h2 className="text-3xl font-semibold tracking-tight sm:text-4xl">The balancer mostly says no</h2>
              <Prose>
                <p>
                  K3 keeps expert load even with a learned bias per expert, added only when choosing the top 16. That bias is
                  a record of training-time load. Early layers keep it within about ±0.15. Deep layers push popular experts
                  down hard, to −0.76 at layer 90, while the biggest boost anywhere is 0.15.
                </p>
              </Prose>
            </div>
          </div>
        </Section>

        <Section>
          <div className="flex flex-col gap-6">
            <h2 className="text-3xl font-semibold tracking-tight sm:text-4xl">Smaller things in the checkpoint</h2>
            <ul className="flex max-w-[65ch] list-disc flex-col gap-3 pl-5 text-[17px] leading-relaxed text-foreground/85">
              <li>
                Every KDA layer stores <Code>A_log</Code> with 128 values for 96 heads. The kernel reads the first 96; the
                last 32 are exactly zero.
              </li>
              <li>
                The attention-residual queries cluster in blocks of 12, matching <Code>attn_res_block_size</Code>: cosine{" "}
                {findings.attn_res.within_block.toFixed(2)} within a block, {findings.attn_res.across_block.toFixed(2)} across.
              </li>
              <li>Input and output embeddings are untied (mean row cosine {findings.untrained.embed_unembed_cos}).</li>
            </ul>
          </div>
        </Section>

        <div className="border-y border-border/60 bg-card/40">
          <Section id="method">
            <div className="flex flex-col gap-8">
              <h2 className="text-3xl font-semibold tracking-tight sm:text-4xl">How it works</h2>
              <Prose>
                <p>
                  A safetensors file starts with a JSON header that lists every tensor's byte offsets. Hugging Face serves
                  files with HTTP range support, so any single tensor in the 1.56 TB checkpoint can be fetched on its own.
                </p>
              </Prose>
              <ol className="flex max-w-[65ch] list-decimal flex-col gap-3 pl-5 text-[17px] leading-relaxed text-foreground/85">
                <li>Read 8 bytes for the header length, then the header, from each of the 96 shards.</li>
                <li>Request only the byte range of the tensors an experiment needs, coalesced into ~96 MB chunks.</li>
                <li>Decode MXFP4 experts: two 4-bit E2M1 values per byte, one power-of-two scale per 32 weights.</li>
                <li>Run the analysis on the GPU and stream the next chunk while it computes.</li>
              </ol>
              <pre className="overflow-x-auto rounded-lg border border-border bg-background p-4 font-mono text-[13px] leading-relaxed text-foreground/85">
{`n = struct.unpack("<Q", get_range(shard, 0, 7))[0]
header = json.loads(get_range(shard, 8, 7 + n))
start, end = header[name]["data_offsets"]
tensor = get_range(shard, 8 + n + start, 8 + n + end - 1)`}
              </pre>
              <Prose>
                <p>
                  Both runs together took {Math.round(findings.run.gpu_minutes)} minutes on a free Kaggle T4. Everything, from
                  the Kaggle scripts to the raw outputs and the analysis, is in the repository.
                </p>
              </Prose>
              <div className="flex flex-wrap gap-3">
                <a href={REPO_URL} className="rounded-md bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90">
                  Open the repository
                </a>
                <a href={MODEL_URL} className="rounded-md border border-border px-4 py-2.5 text-sm font-medium hover:border-foreground/40">
                  Kimi K3 on Hugging Face
                </a>
              </div>
            </div>
          </Section>
        </div>

        <Section>
          <div className="flex flex-col gap-6">
            <h2 className="text-2xl font-semibold tracking-tight">What didn't work</h2>
            <ul className="flex max-w-[65ch] list-disc flex-col gap-3 pl-5 text-[16px] leading-relaxed text-foreground/80">
              <li>
                Routing raw token embeddings through each router, to map which tokens own which expert, only means something
                at layer 1. Deeper routers see contextual states, and on raw embeddings the balancing bias made load less even
                in 91 of 92 layers. Those maps are in the repo but not in these findings.
              </li>
              <li>The glitch-token method from prior work flagged frequent tokens like spaces and commas, because K3's reserved rows never collapsed.</li>
              <li>The first Kaggle run ran out of memory on the T4's 14.5 GB. Chunked matrix math fixed it.</li>
            </ul>
          </div>
        </Section>
      </main>

      <footer className="border-t border-border/60">
        <div className={`mx-auto flex max-w-6xl flex-col gap-3 py-10 text-sm text-muted-foreground ${gutter} sm:flex-row sm:justify-between`}>
          <p>Code and findings MIT licensed. Kimi K3 weights are © Moonshot AI under the Kimi K3 License; none are redistributed here.</p>
          <p>
            Charts by <a className="underline hover:text-foreground" href="https://tripwire.sh/dither-kit">Dither Kit</a>, hero by{" "}
            <a className="underline hover:text-foreground" href="https://dithercn.dev">dithercn</a>.
          </p>
        </div>
      </footer>
    </div>
  )
}
