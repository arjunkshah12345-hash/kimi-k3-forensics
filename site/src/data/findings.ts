import raw from "./findings.json"

export type GalleryItem = { layer: number; expert: number; promotes: string[]; opposite: string[] }

export const findings = raw as typeof raw & {
  writers: typeof raw.writers & { gallery: GalleryItem[] }
}

export const REPO_URL = "https://github.com/arjunkshah12345-hash/kimi-k3-forensics"
export const MODEL_URL = "https://huggingface.co/moonshotai/Kimi-K3"

/** Render a raw tokenizer string so whitespace and tabs stay visible. */
export function showToken(t: string) {
  return t.replace(/\t/g, "⇥").replace(/\n/g, "↵").replace(/^ /, "␣")
}
