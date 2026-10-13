// Client-side dice roller. The player types `/roll d20+3` (or `/r d20+3`)
// anywhere in their turn; before sending we parse the spec, roll the
// dice in JavaScript, and substitute the spec with a stable
// `[ROLL d20+3 = 17 (14+3)]` marker. The DM persona prompt teaches the
// LLM to read that marker as the deterministic outcome of the action.
//
// Doing this client-side keeps it transparent (the player sees their
// own roll in the transcript, before the DM even sees it) and removes
// a class of LLM failure modes — local LLMs are surprisingly bad at
// rolling fair dice or even doing the addition.

export interface RollResult {
  spec: string; // canonicalised, e.g. "2d6+1"
  rolls: number[]; // individual die results
  modifier: number;
  total: number;
  rendered: string; // the `[ROLL ...]` marker that goes into the message
}

// `[/]roll <count>d<sides>[+|-<mod>]` — `count` and the modifier are
// optional; defaults to 1 die and 0 modifier. Allowed dice sides are
// kept permissive (any positive integer) since exotic d100 / d3 rolls
// come up. Case-insensitive on the `d`.
const DICE_SPEC_RE = /\b\/(?:roll|r)\s+(\d+)?d(\d+)(?:\s*([+-])\s*(\d+))?\b/gi;

const MAX_DICE_PER_ROLL = 100; // belt-and-braces against `/roll 999999d20`
const MAX_SIDES = 1000;

export function replaceRollsInline(text: string): { text: string; rolls: RollResult[] } {
  const rolls: RollResult[] = [];
  const replaced = text.replace(
    DICE_SPEC_RE,
    (_match, countStr: string | undefined, sidesStr: string, sign: string | undefined, modStr: string | undefined) => {
      const count = clampInt(parseInt(countStr ?? '1', 10), 1, MAX_DICE_PER_ROLL);
      const sides = clampInt(parseInt(sidesStr, 10), 2, MAX_SIDES);
      const rawMod = modStr ? parseInt(modStr, 10) : 0;
      const modifier = sign === '-' ? -rawMod : rawMod;
      const result = rollSpec(count, sides, modifier);
      rolls.push(result);
      return result.rendered;
    }
  );
  return { text: replaced, rolls };
}

function rollSpec(count: number, sides: number, modifier: number): RollResult {
  const rolls: number[] = [];
  for (let i = 0; i < count; i++) {
    rolls.push(secureRandomInt(1, sides));
  }
  const raw = rolls.reduce((a, b) => a + b, 0);
  const total = raw + modifier;
  const spec = formatSpec(count, sides, modifier);
  const rendered = formatMarker(spec, total, raw, modifier, rolls);
  return { spec, rolls, modifier, total, rendered };
}

function formatSpec(count: number, sides: number, modifier: number): string {
  const base = `${count}d${sides}`;
  if (modifier === 0) return base;
  return modifier > 0 ? `${base}+${modifier}` : `${base}${modifier}`;
}

function formatMarker(
  spec: string,
  total: number,
  raw: number,
  modifier: number,
  rolls: number[]
): string {
  // The breakdown shows individual dice if there are multiple, plus the
  // modifier. Single d20 with no mod just shows the total: `[ROLL d20 = 14]`.
  if (rolls.length === 1 && modifier === 0) {
    return `[ROLL ${spec} = ${total}]`;
  }
  const diceBreakdown = rolls.length > 1 ? `${rolls.join('+')}` : `${raw}`;
  if (modifier === 0) {
    return `[ROLL ${spec} = ${total} (${diceBreakdown})]`;
  }
  const modPart = modifier > 0 ? `+${modifier}` : `${modifier}`;
  return `[ROLL ${spec} = ${total} (${diceBreakdown}${modPart})]`;
}

function secureRandomInt(min: number, max: number): number {
  // crypto.getRandomValues is unbiased and works in every modern
  // browser; we use it so a busy session can't predict its own dice.
  const range = max - min + 1;
  const maxUint = 0x100000000;
  const cutoff = maxUint - (maxUint % range);
  const buf = new Uint32Array(1);
  // Loop until we sample below the rejection cutoff — eliminates the
  // modulo bias even though it's tiny for our range sizes.
  for (;;) {
    crypto.getRandomValues(buf);
    if (buf[0] < cutoff) {
      return min + (buf[0] % range);
    }
  }
}

function clampInt(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.max(min, Math.min(max, Math.floor(value)));
}
