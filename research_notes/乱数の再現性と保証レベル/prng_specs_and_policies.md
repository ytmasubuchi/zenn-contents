# PRNG Specifications and Stability Policies in the Python/NumPy Scientific Stack

Research notes compiled 2026-10-14. All web sources accessed 2026-10-14 unless otherwise noted. Goal: separate three layers for each "set a seed, get the same numbers" claim —
(A) **guaranteed by a published algorithm specification** (any correct implementation must match),
(B) **guaranteed by a library's written policy** (a promise the library makes about its own releases, not about the algorithm itself), and
(C) **an implementation detail** that works today but carries no guarantee and has in fact changed before.

---

## Executive synthesis (for the blog author)

| Claim | Layer | Evidence |
|---|---|---|
| MT19937's twist+tempering step is a fully specified deterministic function of its 624-word state | (A) spec | Matsumoto & Nishimura 1998 TOMACS paper + reference code `mt19937ar.c` |
| "Same seed int → same MT19937 stream" across CPython `random`, NumPy legacy `RandomState`, and NumPy's new `Generator(MT19937(...))` | **False in general** — each uses a different, non-interchangeable seed→state mapping (`init_by_array` over byte-split int vs. NumPy's own `mt19937_seed` vs. `SeedSequence`) | Source-level diff, see MT19937 section below |
| PCG is "one algorithm" | **False** — PCG is an explicitly named *family*; NumPy pins down one member, "PCG XSL RR 128/64" | O'Neill's paper title; NumPy PCG64 docs |
| NumPy's PCG64 default bit generator changed | **Only the recommendation changed, not the shipped default** — PCG64DXSM (added NumPy 1.21.0, 2021) is recommended only for massively-parallel (millions-of-streams) use; `default_rng()` still defaults to plain PCG64 as of current stable docs | numpy.org upgrading-pcg64 page, generator.html |
| Philox/Threefry output is a pure function of (key, counter), stateless, and portable across CPU/GPU by design | (A) spec, empirically demonstrated in the original paper on 3 different chips | Salmon et al. SC11 2011 |
| NumPy ships Threefry | **False** — NumPy ships only `Philox`; Threefry is used elsewhere (e.g., JAX) and by the third-party `randomgen` package | numpy.org bit_generators index |
| Legacy `RandomState` stream is frozen version-to-version forever | (B) **explicit written policy**, NEP 19 + numpy.org/doc/.../compatibility.html | Quoted verbatim below |
| `Generator`'s stream (bit generator + distributions) is guaranteed stable version-to-version | (B) **explicitly NOT guaranteed** beyond narrow same-build/same-machine conditions; NumPy reserves the right to break it on X.Y (feature) releases, and documents concrete cases where it did | NEP 19 + compatibility.html + 1.19.0/1.22.0 release notes |
| `RandomState.standard_normal` and `Generator.standard_normal` give the same numbers for the same seed | **False by design** — legacy uses polar/Box-Muller, Generator uses 256-step ziggurat; NumPy docs state this explicitly | `new-or-different.html`, source `legacy-distributions.c` vs `distributions.c` |
| scikit-learn guarantees identical `fit()` results for a fixed `random_state` across versions | **Not guaranteed** — scikit-learn's own glossary classifies stream-changing bug fixes as an accepted exception to its (informal) backward-compatibility contract | scikit-learn Glossary, "backwards compatibility" entry, quoted verbatim below |
| `SeedSequence` exists only to turn an int into bits | **Undersells it** — it is specifically an avalanche-hash mixing layer designed to defeat (1) low-quality/low-entropy seeds and (2) correlated streams from sequential seeds or careless parallel spawning | numpy.org parallel.html, `bit_generator.pyx` source |

---

## MT19937: what the 1998 paper specifies, and is the stream reproducible across implementations/languages?

### Takeaway
The Matsumoto–Nishimura 1998 TOMACS paper fully specifies a deterministic finite-state machine — a 624×32-bit state array, a linear "twist" recurrence with period 2^19937−1, and a four-step tempering transform — such that two implementations with an *identical* internal state will emit bit-identical output words forever after. But that guarantee only covers the twist+tempering core. Seeding (turning a user-supplied integer into the initial 624-word state) is a separate, less-rigorously-specified layer, and it is exactly here that cross-implementation reproducibility breaks down in practice.

### Cited Findings
- Paper: Matsumoto & Nishimura, "Mersenne twister: a 623-dimensionally equidistributed uniform pseudo-random number generator," ACM TOMACS, Vol. 8, No. 1, 1998. Abstract: the generator has "a super astronomical period of 2^19937 − 1 and 623-dimensional equidistribution up to 32-bit accuracy, while using a working area of only 624 words." — [ACM DL](https://dl.acm.org/doi/10.1145/272991.272995)
- Reference C source: `N 624`, `M 397` (state = 624 32-bit words), `MATRIX_A 0x9908b0dfUL` (twist constant). Header: "A C-program for MT19937, with initialization improved 2002/1/26. Coded by Takuji Nishimura and Makoto Matsumoto." — [mt19937ar.c](https://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/MT2002/CODES/mt19937ar.c)
- Tempering transform constants (consistent with reference code): u=11, d=0xFFFFFFFF, s=7, b=0x9D2C5680, t=15, c=0xEFC60000, l=18, applied as `y = x ^ ((x>>u)&d); y ^= (y<<s)&b; y ^= (y<<t)&c; z = y ^ (y>>l)`. — [Wikipedia: Mersenne Twister](https://en.wikipedia.org/wiki/Mersenne_Twister)
- Author's own project page: MT is "A very fast random number generator of period 2^19937−1," explicitly "not cryptographically secure" — i.e. fully deterministic/predictable from state, which is why it's unsuitable for crypto. The 2002-dated `mt19937ar.c` is listed as the "Newest" reference code. — [math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html](https://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/emt.html)
- CPython's own docs independently corroborate the spec parameters: "Python uses the Mersenne Twister as the core generator. It produces 53-bit precision floats and has a period of 2**19937-1." — [docs.python.org/3/library/random.html](https://docs.python.org/3/library/random.html)
- NumPy's modern (non-legacy) `MT19937` bit generator page states its own version-to-version compatibility promise: "`MT19937` makes a guarantee that a fixed seed will always produce the same random integer stream" — but this is a NumPy-version-to-version promise about NumPy's own seeding path (via `SeedSequence`), **not** a claim of cross-library/cross-language bit-identity. — [numpy.org MT19937 docs](https://numpy.org/doc/stable/reference/random/bit_generators/mt19937.html)

### Inferences
- "MT19937" as a name is not, by itself, enough information to reproduce a stream — you additionally need to know exactly which seeding convention was used (this is the crux of the CPython-vs-NumPy divergence documented below).

### Gaps
- Could not retrieve the ACM-hosted full PDF text directly (paywalled); claims about the paper's exact proof content rely on the abstract, the reference code, and secondary descriptions (Wikipedia), not a direct read of the paper body.
- Could not confirm from the 1998 paper itself whether it specifies seeding at all, or treats it as out of scope — `init_genrand` (simple scalar seeding) predates the 2002 `init_by_array` addition, suggesting seeding was treated as a secondary, evolving concern rather than part of the core 1998 specification.

---

## MT19937 seeding: init_genrand vs. init_by_array, and does CPython `random` match NumPy's legacy `RandomState`?

### Takeaway
The core twist/tempering algorithm is portable; **seeding is not**. The reference code offers two seeding routines — `init_genrand(seed)` (single 32-bit int, original) and `init_by_array(key[], key_length)` (array-based, added 2002) — but CPython and NumPy both build their own, mutually incompatible seed→state mappings on top of this, and NumPy's legacy `RandomState` additionally diverges from the reference code itself. **CPython's `random.seed(42)`, NumPy's `RandomState(42)`, and NumPy's `Generator(MT19937(42))` all produce different streams**, even though all three nominally "use MT19937."

### Cited Findings
- `init_genrand`: seeds via the recurrence `mt[i] = 1812433253UL * (mt[i-1] ^ (mt[i-1] >> 30)) + i`, computed sequentially from a single 32-bit seed. — [mt19937ar.c](https://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/MT2002/CODES/mt19937ar.c)
- `init_by_array`: "first calls `init_genrand` with 19650218UL, then performs two mixing loops" using multipliers 1664525UL and 1566083941UL, "ensuring MSB is 1; assuring non-zero initial array." Added "2002/1/26" — i.e., a later addition to the original 1998 algorithm, not part of the TOMACS paper. — [mt19937ar.c](https://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/MT2002/CODES/mt19937ar.c); confirmed as "Newest at present, 2004/3" on the [author's index page](https://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/emt.html)
- **CPython** (`Modules/_randommodule.c`): for an int seed, "use its absolute value," then "split n into 32-bit chunks, from the right" via `_PyLong_AsByteArray`, and feed the resulting word array to `init_by_array(self, key, keyused)`. String/bytes seeds are SHA-512-hashed first, then the digest is treated as an int and fed through the same path. — [cpython/Modules/_randommodule.c](https://github.com/python/cpython/blob/main/Modules/_randommodule.c)
  - `random.seed(a=None, version=2)` docs: "If a is an int, its absolute value is used directly." "With version 2 (the default), a str, bytes, or bytearray object gets converted to an int and all of its bits are used." "Changed in version 3.2: Moved to the version 2 scheme which uses all of the bits in a string seed." — [docs.python.org/3/library/random.html#random.seed](https://docs.python.org/3/library/random.html)
  - So CPython always uses `init_by_array`, even for a tiny scalar int (treated as a 1-word "array").
- **NumPy legacy `RandomState`**: docs explicitly flag the divergence from the modern path: "note that its default seeding [for the `MT19937` BitGenerator used with `Generator`] uses `SeedSequence` rather than the legacy seeding algorithm. `RandomState` will use the legacy seeding algorithm." — [NumPy Legacy Random Generation docs](https://numpy.org/doc/stable/reference/random/legacy.html)
  - Source (`_mt19937.pyx`): for a scalar int seed, calls `mt19937_seed(&self.rng_state, seed)`; only for an array-like seed does it call `mt19937_init_by_array`. — [numpy/random/_mt19937.pyx](https://github.com/numpy/numpy/blob/main/numpy/random/_mt19937.pyx)
  - `mt19937_seed()` itself (`mt19937.c`): `state->key[pos] = seed; seed = (1812433253UL * (seed ^ (seed >> 30)) + pos + 1) & 0xffffffffUL;` for each of 624 positions. This reuses the `1812433253` constant from `init_genrand` but has **different data flow**: NumPy assigns the *current* running seed to `state->key[pos]` *before* advancing it, whereas the reference `init_genrand` computes `mt[pos]` from the *already-advanced* previous value. These are not the same algorithm. — [numpy/random/src/mt19937/mt19937.c](https://github.com/numpy/numpy/blob/main/numpy/random/src/mt19937/mt19937.c)
  - So NumPy legacy scalar-int seeding matches **neither** the reference `init_genrand` **nor** the reference `init_by_array` — it is NumPy's own, third variant.
- NumPy's modern `MT19937` bit generator (used by `Generator`) instead "processed by `SeedSequence` to fill the whole state. The first element is reset such that only its most significant bit is set" — a fourth, unrelated seeding construction. — [NumPy MT19937 docs](https://numpy.org/doc/stable/reference/random/bit_generators/mt19937.html)
- Secondary/community corroboration (lower authority): a cross-language MT19937 reproducibility project notes that getting identical streams across Python/MATLAB/C++/Octave requires non-default, careful alignment work, and separately notes "transitioning from NumPy's legacy `numpy.random` tools to newer generators can be challenging due to differences in underlying algorithms and states." — [wjassim/CrossLangRandomSeeds](https://github.com/wjassim/CrossLangRandomSeeds)

### Inferences
- "CPython's `random` and NumPy's `RandomState`, given the same seed integer, produce different output streams" is a direct, verifiable consequence of comparing the two projects' seeding source code (different data flow into the same 624-word state slots) — it does not require running code to confirm, though no single official doc states this comparison explicitly side by side (see Gaps).

### Gaps
- No single official NumPy or CPython document explicitly states "CPython `random.seed(42)` and NumPy `RandomState(42)` do NOT produce the same stream" — this is an inference from comparing seeding source code directly, not a quoted authoritative claim.
- Could not verify whether CPython's int-seeding byte-splitting logic itself has changed across CPython versions (seed *handling for str/bytes* changed in 3.2, and accepted seed *types* changed in 3.11, per the docs, but the int path's stability wasn't separately confirmed).
- No documented rationale found for *why* NumPy's `mt19937_seed` data-flow diverges from the reference `init_genrand` — whether intentional (e.g. better avalanche behavior for small seeds) or a historical implementation accident.

---

## PCG64: what does O'Neill's PCG specification fix, and what does NumPy add?

### Takeaway
O'Neill's PCG work is explicitly a *family* of algorithms, not one fixed output sequence. It specifies a general two-stage structure (LCG state advance, then an output-permutation function applied to — without altering — that state) but leaves the specific permutation function, the internal state width (8–128 bits), and the multiplier/increment constants configurable. "PCG" alone therefore does not pin down a bit-reproducible sequence; NumPy's own docs make this explicit by naming the exact family member they implement.

### Cited Findings
- Paper title itself asserts the family framing: M.E. O'Neill, "PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for Random Number Generation." — [PDF](https://www.pcg-random.org/pdf/toms-oneill-pcg-family-v1.02.pdf), mirrored at [cs.hmc.edu](https://www.cs.hmc.edu/tr/hmc-cs-2014-0905.pdf)
- pcg-random.org homepage tagline: "PCG, A Family of Better Random Number Generators." — [pcg-random.org](https://www.pcg-random.org/index.html)
- (Secondary, Wikipedia, consistent with primary framing but not independently verified against the PDF): "A permuted congruential generator (PCG) ... applies an output permutation function to improve the statistical properties of a modulo-2^n linear congruential generator (LCG)." "The core LCG is defined for widths from 8 to 128 bits" and "the additive constant in the LCG can be varied to produce different streams." Named output-permutation variants include XSH-RR ("the generator recommended for most users"), XSH-RS, XSL-RR, RXS-M-XS (different speed/strength tradeoffs). — [Wikipedia: Permuted congruential generator](https://en.wikipedia.org/wiki/Permuted_congruential_generator)
- **NumPy explicitly names its exact family member**, confirming "PCG" alone is underspecified: "The specific member of the PCG family that we use is PCG XSL RR 128/64 as described in the paper." State = "2 unsigned 128-bit values... the state of the PRNG ... advanced by a linear congruential generator (LCG)" plus "a fixed odd increment used in the LCG." — [numpy.org PCG64 docs](https://numpy.org/doc/stable/reference/random/bit_generators/pcg64.html)
- NumPy's C header defines the 128-bit default multiplier as `PCG_DEFAULT_MULTIPLIER_HIGH 2549297995355413924ULL` / `PCG_DEFAULT_MULTIPLIER_LOW 4865540595714422341ULL` (= `0x2360ed051fc65da44385df649fccf645`), matching O'Neill's own canonical 128-bit default multiplier (no code comment flags any numpy-specific deviation for the base PCG64 path). A separate `PCG_CHEAP_MULTIPLIER_128 (0xda942042e4dd58b5ULL)` is used only by the PCG64DXSM path. — [numpy/random/src/pcg64/pcg64.h](https://github.com/numpy/numpy/blob/main/numpy/random/src/pcg64/pcg64.h)

### Inferences
- The task brief's premise that NumPy's base PCG64 multiplier/output function silently diverges from O'Neill's reference was **not substantiated** — the one confirmed divergence from upstream is narrower and specific to PCG64DXSM's seeding step (see next section).

### Gaps
- Could not directly fetch O'Neill's own paper/pcg-random.org specification pages verbatim (repeated WebFetch timeouts); the XSH-RR/XSL-RR/RXS-M-XS mnemonic descriptions come from secondary sources (Wikipedia, HMC summaries), internally consistent but not independently checked against the primary PDF text.
- The 8–128-bit configurable state-width range is Wikipedia's paraphrase, not independently verified against the primary PDF.

---

## PCG64 vs PCG64DXSM, and NumPy's default/recommended bit generator history

### Takeaway
NumPy added **PCG64DXSM in NumPy 1.21.0 (2021)** — not 1.25 — to fix a real statistical weakness in PCG64 under massively-parallel stream use. As of current stable docs (directly verified by fetch), `default_rng()`'s actual shipped default bit generator is **still plain PCG64**; NumPy's *recommendation* for large-scale parallel use has shifted toward PCG64DXSM, but even that recommendation is narrower than "always prefer DXSM" — it is explicitly scoped to users spawning millions of parallel streams, not to new code in general. NumPy has stated an intent for PCG64DXSM to "eventually" become the default; this had not happened as of the access date.

### Cited Findings
- **The weakness PCG64DXSM fixes**: "Uses of the PCG64 BitGenerator in a massively-parallel context have been shown to have statistical weaknesses that were not apparent at the first release in numpy 1.17. However, most users will never observe this weakness and are safe to continue to use PCG64." Mechanism: "advancing the LCG state by powers of two leaves the lower N bits identical"; empirically, "if one shares the lower 58 bits of state and shares an increment, then the pair of streams, when interleaved, will fail PractRand in a reasonable amount of time, after drawing a few gigabytes of data." Collision risk becomes material at roughly 2^29 (~half a billion) simultaneously-drawn parallel streams. — [numpy.org upgrading-pcg64.html](https://numpy.org/doc/stable/reference/random/upgrading-pcg64.html) (fetched directly, verbatim, accessed 2026-10-14)
- **The fix**: "The DXSM output function adopts a 'xorshift-multiply' construction used in strong integer hashes that has much better avalanche properties than the XSL-RR output function," converting near-identical-state streams into "practically-independent output streams." PCG64DXSM "differs from PCG64 in that it uses the stronger DXSM output function, a 64-bit 'cheap multiplier' in the LCG, and outputs from the state before advancing it rather than advance-then-output." — same source
- **Exact current recommendation wording (verbatim, directly fetched 2026-10-14)**: the page does **not** broadly recommend PCG64DXSM for all new code. It scopes the recommendation to very-large-scale parallel use ("very large numbers of parallel streams, in the millions"): "Such use cases should consider using `PCG64DXSM` explicitly or another modern `BitGenerator` like `SFC64` or `Philox`." — [numpy.org upgrading-pcg64.html](https://numpy.org/doc/stable/reference/random/upgrading-pcg64.html)
- **Stated intent, not yet realized (verbatim, directly fetched 2026-10-14)**: "We have introduced a new `PCG64DXSM` `BitGenerator` that will eventually become the new default `BitGenerator` implementation used by `default_rng` in future releases." — same source
- **Default as shipped today**: "The default BitGenerator used by `Generator` is `PCG64`." `default_rng()` is documented as "Construct a new Generator with the default BitGenerator (PCG64)." — [numpy.org generator.html](https://numpy.org/doc/stable/reference/random/generator.html)
- **Version introduced — NumPy 1.21.0, not 1.25**: confirmed via the implementing PR "ENH: Add PCG64DXSM BitGenerator" (numpy/numpy#18906, by rkern), which closed issue #16313 ("The PCG implementation provided by Numpy has significant, dangerous self-correlation") and merged into `main` on 2021-05-05, targeting the 1.21 release cycle. — [github.com/numpy/numpy/pull/18906](https://github.com/numpy/numpy/pull/18906)
- **Confirmed narrow divergence from upstream PCG, specific to PCG64DXSM's seeding only**: GitHub issue #22472, "PCG64DXSM initial state doesn't get advanced with a 64-bit 'cheap multiplier' in the LCG," reports NumPy's PCG64DXSM seeding call chain (`pcg64_set_seed` → `pcg_setseq_128_srandom_r` → `pcg_setseq_128_step_r`) uses the default 128-bit multiplier during seeding rather than the 64-bit "cheap multiplier" that O'Neill's own PCG64DXSM reference uses at that step. Per the discussion, this was left unresolved/low-priority because (a) NumPy's stream-compatibility policy blocks a silent in-place fix, and (b) seeding-step precision is considered immaterial given NumPy's `SeedSequence`-based seeding. — [github.com/numpy/numpy/issues/22472](https://github.com/numpy/numpy/issues/22472)
- A related, broader claim of divergence (issue #28289, "pcg64_dxsm random number generator has different implementation in comparison to reference") was investigated and closed: the core DXSM output step (128-bit state × 64-bit cheap multiplier) is confirmed *intentional*, matching upstream PCG's `cm_setseq_dxsm_128_64` variant — "The cheap multiplier is important for performance reasons; it makes up for the more expensive DXSM scrambling output function. This is the variant that received the most PractRand testing." The broader claim was attributed to comparing against the wrong reference variant. — [github.com/numpy/numpy/issues/28289](https://github.com/numpy/numpy/issues/28289)
- **Why the default can legally change without breaking any promise — this is itself a documented policy choice, not an accident**: NumPy's compatibility policy explicitly states "`default_rng()` is allowed to change which BitGenerator it uses by default," and that stream-compatibility "can be broken to introduce new features or improve performance in `Generator` or `default_rng`, but only in feature releases (X.Y), never in bugfix releases (X.Y.Z)." — [numpy.org compatibility.html](https://numpy.org/doc/stable/reference/random/compatibility.html)

### Inferences
- The PCG64 → PCG64DXSM episode is a clean worked example of the report's core theme: the *algorithm family* (PCG) is specified by O'Neill; *which exact variant* NumPy ships, and *when/whether* it's the default, is purely NumPy policy — and NumPy has explicitly reserved the right to change it, which it has partly exercised (new option added, old default kept) rather than fully exercised (default not yet flipped).

### Gaps
- Could not independently fetch O'Neill's own primary-source blog post specifically describing the DXSM design (only referenced indirectly via a third-party blog, dotat.at, whose full page also failed to fetch) — treat "DXSM = double xorshift multiply" and its rationale as community-sourced, not O'Neill-primary-verified in this research pass.
- No evidence found (nor expected, since 2026-10-14 is the access date) of any NumPy release after current stable docs that has since flipped the actual default from PCG64 to PCG64DXSM.

---

## Philox/Threefry (Random123): what does the SC11 paper specify, and is it portable across implementations/CPU/GPU?

### Takeaway
Salmon, Moraes, Dror & Shaw's SC11 paper ("Parallel Random Numbers: As Easy as 1, 2, 3") defines counter-based PRNGs as **stateless keyed bijections** — pure functions `x_n = b_k(n)` of (key, counter) built entirely from fixed-width integer arithmetic (multiply/add/XOR/rotate), with no carried state and no floating point. Because the spec is pure fixed-width integer math, it is in principle bit-reproducible by any correct implementation on any platform, and the paper itself empirically demonstrates this by running one unmodified source file on a CPU and two different GPU architectures. However, "same algorithm name" does not automatically mean "same output" — round count, bit-width variant, and deliberate implementation extensions can and do break bit-compatibility between generators that both call themselves "Threefry" or "Philox." NumPy ships Philox only; Threefry is not part of NumPy core.

### Cited Findings
- Paper: Salmon, Moraes, Dror, Shaw, "Parallel Random Numbers: As Easy as 1, 2, 3," SC11 2011, ACM, Best Paper Award. DOI: [dl.acm.org/doi/10.1145/2063384.2063405](https://dl.acm.org/doi/10.1145/2063384.2063405); full text: [thesalmons.org/john/random123/papers/random123sc11.pdf](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf)
- Stateless pure-function definition: formalized as `x_n = b_k(n)` — "if b is a 'bijection' ... then the period of the generator ... is 2^p" — motivated because counter-based designs "require little or no memory for state," letting any process compute any slice of the sequence independently. — same source
- Threefry derivation: "Threefry-N×W-R follows the same pattern of S and P boxes as Threefish, but with R rounds, and N W-bit inputs and outputs" — i.e. reuses the Threefish/Skein block-cipher mixing function, stripped of the full cipher's key schedule. — same source
- Philox derivation: "Philox is an SP network with an overall structure similar to Threefry, but with an S-box that uses multiplication to provide diffusion" — a new, multiplication-based (not cipher-derived) design. — same source
- Minimum statistically-safe round counts (Crush-resistant per TestU01): "Threefry-2×64-13, Threefry-4×32-12, and Threefry-4×64-12 are all Crush-resistant" and "Philox-4×32-7, Philox-2×64-6, Philox-4×64-7" are the minimal Crush-resistant configurations, with extra rounds recommended for safety margin. — same source
  - (Secondary, not independently primary-confirmed on numpy.org): NumPy's default is commonly cited elsewhere as "Philox4x64-10" (10 rounds); treat the exact numpy default round count as secondary-sourced only.
- **Cross-platform portability — design goal, empirically demonstrated**: "A single set of header files implements the API in C, C++98, C++0x, CUDA and OpenCL environments," and "the same source code was used for counter-based PRNGs on all platforms," tested on a 3.07 GHz Intel Xeon X5667 CPU, an NVIDIA GeForce GTX580 GPU, and an AMD Radeon HD6970 GPU, concluding the designs "are portable, indistinguishable in terms of statistical quality ... and faster than any Crush-resistant conventional PRNG." — same source
- **Independent (vendor) confirmation that the counter-based core, but not downstream transforms, is cross-platform-identical**: MathWorks/MATLAB documents that setting the same algorithm (Threefry or Philox) and seed via `rng` (CPU) and `gpurng` (GPU) yields identical *uniform* draws, but warns non-uniform transforms are a separate concern: "The only transform supported on both the CPU and GPU is 'Inversion'" (CPU otherwise defaults to Ziggurat, GPU to Box-Muller). — [mathworks.com random-number-streams-on-a-GPU](https://www.mathworks.com/help/parallel-computing/random-number-streams-on-a-GPU.html)
  - This is directly relevant to the report's core distinction: the raw counter-based primitive's cross-platform bit-reproducibility is a property of the published algorithm, but *which* non-uniform sampling transform sits on top is a separate library/implementation choice, not part of the Philox/Threefry spec itself.
- **NumPy ships Philox, not Threefry**: the bit-generator index lists PCG64 (default), PCG64DXSM, MT19937, Philox, SFC64 — no Threefry anywhere. — [numpy.org bit_generators/index.html](https://numpy.org/doc/stable/reference/random/bit_generators/index.html)
  - NumPy's Philox page: "a 64-bit PRNG that uses a counter-based design based on weaker (and faster) versions of cryptographic functions," state = 256-bit counter (4×uint64) + 128-bit key (2×uint64), citing Salmon et al. SC11 as its sole reference. NumPy's own compatibility statement: "`Philox` makes a guarantee that a fixed `seed` will always produce the same random integer stream" — a NumPy-version-to-version promise, not an explicit claim of bit-identity with other libraries' Philox implementations. — [numpy.org philox.html](https://numpy.org/doc/stable/reference/random/bit_generators/philox.html)
  - Threefry is used by JAX instead, and by the third-party NumPy-API-compatible package `randomgen`. — [jax docs](https://docs.jax.dev); [github.com/bashtage/randomgen](https://github.com/bashtage/randomgen)
- **Concrete divergence example — same algorithm name, different implementation detail, different output**: JAX's `jax_threefry_partitionable` flag (default `True` as of JAX 0.5.0) deliberately changes how `threefry2x32` is computed internally to make it auto-parallelizable across devices — meaning JAX's "Threefry" no longer matches the original Random123 reference algorithm's output bit-for-bit for the same key/counter, by design. — [JAX GitHub discussion #18480](https://github.com/jax-ml/jax/discussions/18480)
- An ISO C++ Library Working Group issue, "Issue with Philox algorithm specification" (LWG 4134), surfaced during the C++ standardization proposal (WG21 P2075) to add Philox as a standard RNG engine — found only via search snippet, title and existence only. — [cplusplus.github.io/LWG/issue4134](https://cplusplus.github.io/LWG/issue4134)

### Inferences
- The JAX Threefry example is a strong, citable "spec vs. implementation detail" case study for the blog: the published algorithm is fully deterministic and portable *on paper*, but a library is free to modify its internals (for performance/parallelism reasons) while keeping the same algorithm name, silently breaking cross-library reproducibility.

### Gaps
- Could not independently re-verify every quoted paper fragment against a second copy of the PDF (first WebFetch attempt returned raw PDF bytes; a second attempt succeeded) — treat quote boundaries as very likely accurate but not double-checked.
- No first-person NumPy statement found claiming "Philox output is bit-identical across CPU and GPU" — NumPy only cites the Random123 paper as a reference; NumPy itself ships no GPU backend, so this specific claim may simply be out of scope for NumPy's own docs.
- NumPy's exact default round count ("Philox4x64-10") was not confirmed against a primary NumPy source, only secondary/mirror docs.
- Did not fetch full text of LWG issue 4134 / WG21 P2075 to characterize the specification ambiguity found in Philox — have title only.

---

## SeedSequence: how does NumPy turn a seed into generator state, and what is it designed to protect against?

### Takeaway
`numpy.random.SeedSequence` is NumPy's seed-to-state mixing layer between a user-supplied seed (int, list of ints, or `None`) and a bit generator's actual internal state. It exists specifically to defeat two documented failure modes: (1) **low-quality seeds** (e.g. naively dropping a 32-bit seed into the last word of MT19937 state with the rest zeroed is legal but statistically bad, because MT19937 degrades when its state has too many zero words), and (2) **correlated seeds** (sequential seeds like 12345/12346 would otherwise produce visibly similar streams). It is not described in a dedicated NEP — it ships as part of the same NEP 19 implementation effort and is documented directly in NumPy's reference-guide pages.

### Cited Findings
- Purpose, verbatim: "A naive way to take a 32-bit integer seed would be to just set the last element of the state to the 32-bit seed and leave the rest 0s. This is a valid state for MT19937, but not a good one. The Mersenne Twister algorithm suffers if there are too many 0s." And: without mitigation, "two adjacent 32-bit integer seeds (i.e. `12345` and `12346`) would produce very similar streams." — [numpy.org parallel.html](https://numpy.org/doc/stable/reference/random/parallel.html)
- Mixing mechanism, verbatim: "SeedSequence avoids these problems by using successions of integer hashes with good avalanche properties to ensure that flipping any bit in the input has about a 50% chance of flipping any bit in the output," so "two input seeds that are very close to each other will produce initial states that are very far from each other (with very high probability)." — same source
- **Algorithmic lineage — O'Neill, not Steele/Lea/Flood**: NumPy source states the implementation is "derived from Melissa E. O'Neill's C++11 `std::seed_seq` implementation," citing her post "Developing a Seed_Seq Alternative." The code implements `hashmix()` (XOR with a hash constant, multiply, right-shift mixing) and `mix()` (combine two values via `MIX_MULT_L`/`MIX_MULT_R` constants with XOR diffusion), and a `mix_entropy()` step to "Mix all bits together so late bits can affect earlier bits." Default entropy pool size is 4 words = 128 bits. — [numpy/random/bit_generator.pyx](https://raw.githubusercontent.com/numpy/numpy/main/numpy/random/bit_generator.pyx); [O'Neill's post](https://www.pcg-random.org/posts/developing-a-seed_seq-alternative.html)
  - **Correction of a plausible misattribution**: the Steele, Lea & Flood OOPSLA 2014 paper "Fast Splittable Pseudorandom Number Generators" (the SplitMix / `java.util.SplittableRandom` lineage) is **not** cited anywhere in NumPy's SeedSequence docs or source — every primary citation found points to O'Neill's `std::seed_seq`-alternative post instead. These are related-in-spirit but technically distinct "splittable PRNG" lineages; do not state NumPy's SeedSequence is based on SplitMix.
- `spawn()` mechanism, verbatim: "Spawn a number of child SeedSequences by extending the spawn_key" — each child i gets `spawn_key = self.spawn_key + (i,)`, giving it a unique position in a spawning tree, mixed through the same hash to produce independent entropy. "Calling spawn(n) will create n SeedSequences that can be used to seed independent BitGenerators, i.e. for different threads." Collision bound explicitly quantified: "The probability that there is a collision in that pool, pessimistically-estimated, will be about n²×2⁻¹²⁸ where n is the number of streams spawned" — e.g. "~2⁻⁸⁸" for a million (2^20) streams, called "solidly-ignorable," and "if one only does one level of spawning, it is guaranteed that all states will be unique." This lets users "make local decisions about when and how to split up streams without coordination between processes." — [numpy.org SeedSequence docs](https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.SeedSequence.html); [numpy.org parallel.html](https://numpy.org/doc/stable/reference/random/parallel.html)
- **No dedicated NEP**: NEP 19's text (https://numpy.org/neps/nep-0019-rng-policy.html) covers only the BitGenerator/Generator architectural split and the compatibility-guarantee subset of methods — it does **not** mention "SeedSequence" anywhere. A secondary source (scientific-python.org blog) confirms SeedSequence shipped as part of the same 1.17 implementation effort rather than via its own NEP. — [blog.scientific-python.org/numpy/numpy-rng](https://blog.scientific-python.org/numpy/numpy-rng/)

### Inferences
- SeedSequence is a concrete example of a *designed* mitigation sitting between "the user's seed" and "the algorithm's state" — it is neither part of the MT19937/PCG64/Philox algorithm specs themselves, nor a stream-compatibility promise; it's a NumPy-specific engineering layer whose job is explicitly to make naive/sequential seeds behave well.

### Gaps
- A third-party, unverified claim (via a GitHub repo referencing numpy/numpy#32063) suggests concurrent `.spawn()` calls on the same parent `SeedSequence` from multiple threads can race on an internal counter and hand out duplicate spawn keys in a multi-threaded (not multi-process) context — this was not confirmed against the primary numpy/numpy issue tracker and should be treated as unverified.
- A GitHub issue title, "DOC: clarification about what SeedSequence can and cannot do" (numpy/numpy#25778), was found via search snippet only, not fetched/read in full.

---

## NEP 19 "Random Number Generator Policy": exact commitments for RandomState vs. Generator

### Takeaway
NEP 19 (status **Final**, authored by Robert Kern, created 2018-05-24, updated 2019-05-21) is the governing document that splits NumPy's RNG API into two tiers with deliberately different guarantees: legacy `RandomState` gets a strict, effectively-frozen compatibility guarantee (no further modifications, not even correctness bug fixes), while the new `Generator`/bit-generator API is explicitly permitted to break stream-compatibility across feature (X.Y) releases. Current numpy.org docs (`compatibility.html`, `new-or-different.html`) restate and operationalize this same split in the live documentation.

### Cited Findings
- **NEP 19 metadata** (https://numpy.org/neps/nep-0019-rng-policy.html): Author: Robert Kern. "Status: Final." "Type: Standards Track." "Created: 2018-05-24." "Updated: 2019-05-21." "Resolution: https://mail.python.org/pipermail/numpy-discussion/2018-July/078380.html"
- **RandomState's strict, effectively-frozen guarantee — exact text**: "The new PRNG subsystem MUST provide a second, legacy distributions class that uses the same implementations of the distribution methods as the current version of `numpy.random.RandomState`. The methods of this class will have strict stream-compatibility guarantees, even stricter than the current policy. It is intended that this class will no longer be modified, except to keep it working when numpy internals change. All new development should go into the primary distributions class. Bug fixes that change the stream SHALL NOT be made to `RandomState`; instead, buggy distributions should be made to warn when they are buggy. The purpose of `RandomState` will be documented as providing certain fixed functionality for backwards compatibility and stable numbers for the limited purpose of unit testing, and not making whole programs reproducible across numpy versions."
  - Note: the literal words "freeze"/"frozen" do **not** appear in NEP 19's own text — "frozen" is a paraphrase of "no longer be modified" / "SHALL NOT [change the stream]." Attribute the characterization accordingly, not as a direct NEP 19 quote.
- **Generator's explicit license to break stream-compatibility — exact text** (same source): "Second, breaking stream-compatibility in order to introduce new features or improve performance will be allowed with caution. Such changes will be considered features, and as such will be no faster than the standard release cadence of features (i.e. on X.Y releases, never X.Y.Z)."
- **Rationale for dropping the old strict guarantee for Generator, exact text** (same source): "The standard practice now for bit-for-bit reproducible research is to pin all of the versions of code of your software stack... We do not need to maintain stream-compatibility across numpy versions to help them."
- **Current numpy.org compatibility policy page restates the same split, live docs, verbatim** (https://numpy.org/doc/stable/reference/random/compatibility.html):
  > "The main kind of compatibility that we enforce is stream-compatibility from run to run under certain conditions. If you create a Generator with the same BitGenerator, with the same seed, perform the same sequence of method calls with the same arguments, on the same build of numpy, in the same environment, on the same machine, you should get the same stream of numbers." [Note: "these conditions are very strict."]
  > "Breaking stream-compatibility in order to introduce new features or improve performance in Generator or default_rng will be allowed with caution." "Such changes will be considered features... (i.e. on X.Y releases, never X.Y.Z)... Correctness bug fixes that break stream-compatibility can happen on bugfix releases, per usual..."
  > "The legacy RandomState and the associated convenience functions have a stricter version-to-version compatibility guarantee." "There will be no more modifications to RandomState, not even to fix correctness bugs."
  > On cross-machine limits: "There are a number of factors outside of NumPy's control that limit our ability to guarantee much more than this. For example, different CPUs implement floating point arithmetic differently, and this can cause differences in certain edge cases that cascade to the rest of the stream. Generator.multivariate_normal ... uses a matrix decomposition from numpy.linalg. Even on the same platform, a different build of numpy may use a different version of this matrix decomposition algorithm from the LAPACK that it links to, causing Generator.multivariate_normal to return completely different (but equally valid!) results."
  > "`default_rng()` is allowed to change which BitGenerator it uses by default."

### Inferences
- The "same build of numpy, in the same environment, on the same machine" qualifier in the compatibility page is itself load-bearing: even `Generator`'s narrow promise is conditioned on holding the entire software/hardware stack fixed, not just the seed — floating-point and LAPACK/BLAS differences are explicitly called out as legitimate sources of divergence outside that promise.

### Gaps
- Did not independently fetch the July 2018 numpy-discussion mailing-list resolution thread (linked as "Resolution:") to confirm it contains wording beyond what's already quoted from the NEP itself.

---

## NumPy distribution transforms: standard_normal algorithm differences and documented stream breaks

### Takeaway
Legacy `RandomState.standard_normal` and new `Generator.standard_normal` use genuinely **different algorithms** — legacy uses the polar method (a Box-Muller variant), Generator uses a 256-step Ziggurat method — and NumPy's own docs state explicitly that seeds do **not** reproduce the same values across the two APIs for these distributions. This is a designed incompatibility, not a bug. Separately, NumPy's release notes document multiple concrete cases where a bug-fix changed a distribution's (or bit generator's) underlying algorithm and explicitly broke fixed-seed reproducibility release-to-release.

### Cited Findings
- **Legacy algorithm — polar method/Box-Muller**, from source (`legacy_gauss` in `legacy-distributions.c`):
  ```c
  /* Polar method, a more efficient version of the Box-Muller approach. */
  f = sqrt(-2.0 * log(r2) / r2);
  ```
  with `x1, x2` drawn from `2*uniform-1` inside a rejection loop (`r2 >= 1.0 || r2 == 0.0`). — [numpy/random/src/legacy/legacy-distributions.c](https://github.com/numpy/numpy/blob/main/numpy/random/src/legacy/legacy-distributions.c)
- **Generator algorithm — Ziggurat**, from source (`random_standard_normal` in `distributions.c`): uses ziggurat tables (`ziggurat_nor_inv_r`, `ziggurat_nor_r`, `ki_double`, `wi_double`, `fi_double`) with a fast-path comment `/* 99.3% of the time return here */` — the classic Marsaglia/Tsang ziggurat rejection-sampling structure. — [numpy/random/src/distributions/distributions.c](https://github.com/numpy/numpy/blob/main/numpy/random/src/distributions/distributions.c)
- **NumPy docs explicitly state the algorithms differ and seeds don't carry across APIs, exact text** ("What's New or Different" page): "The normal, exponential and gamma generators use 256-step Ziggurat methods which are 2-10 times faster than NumPy's default implementation in standard_normal, standard_exponential or standard_gamma. Because of the change in algorithms, it is not possible to reproduce the exact random values using Generator for these distributions or any distribution method that relies on them." — [numpy.org new-or-different.html](https://numpy.org/doc/stable/reference/random/new-or-different.html)
  - This is the official confirmation that `RandomState(seed).standard_normal(n)` and `default_rng(seed).standard_normal(n)` are documented to produce different numeric values for the same seed.
- **Concrete release-note cases where an algorithm change broke a fixed-seed stream**:
  1. Dirichlet, NumPy 1.19.0: "Changed random variate stream from numpy.random.Generator.dirichlet: A bug in the generation of random variates for the Dirichlet distribution with small 'alpha' values was fixed by using a different algorithm when max(alpha) < 0.1. Because of the change, the stream of variates generated by dirichlet in this case will be different from previous releases." (gh-14924) — [numpy.org 1.19.0-notes.html](https://numpy.org/doc/stable/release/1.19.0-notes.html)
  2. SeedSequence + spawning, NumPy 1.19.0: "SeedSequence with small seeds no longer conflicts with spawning: ... Now, the seed is explicitly 0-padded out to the internal pool size before concatenating the spawn key. Spawned SeedSequences will produce different results than in the previous release. Unspawned SeedSequences will still produce the same results." (gh-16551) — same source
  3. MT19937 jump, NumPy 1.19.0: "Fixed the jumping implementation in MT19937.jumped: This fix changes the stream produced from jumped MT19937 generators. It does not affect the stream produced using RandomState or MT19937 that are directly seeded. ... MT19937.jumped matches ... Makoto Matsumoto's original implementation of the Horner and Sliding Window jump methods." (gh-16153) — same source
  4. float32 uniform-derived variates, NumPy 1.22.0: "There was bug in the generation of 32 bit floating point values from the uniform distribution that would result in the least significant bit of the random variate always being 0. This has been fixed. This change affects the variates produced by the random.Generator methods random, standard_normal, standard_exponential, and standard_gamma, but only when the dtype is specified as numpy.float32." (gh-20314) — [numpy.org 1.22.0-notes.html](https://numpy.org/doc/stable/release/1.22.0-notes.html)
  5. Bit-generator level (not a distribution transform, but the same "algorithm change broke the stream" phenomenon): PCG64 → PCG64DXSM — see dedicated section above for verbatim quotes.
  - Lower-confidence, not independently re-verified: search results also pointed to an earlier (1.16.4/1.17.0-era) "log-of-zero" edge-case bug fix affecting beta, binomial, laplace, logistic, logseries, and multinomial sampling when a 0 is generated in the underlying MT19937 stream — this was not independently fetched/quoted from a primary release-notes page and should be verified before direct quotation.

### Inferences
- The explicit "it is not possible to reproduce the exact random values using Generator for these distributions" sentence is one of the single most quotable lines for the blog post's core argument: NumPy itself states, in its own docs, that the raw bit-generator stream being reproducible does not imply the downstream distribution values are.

### Gaps
- Did not independently verify the 1.16.4/1.17.0 "log-of-zero" stream-break claim against a primary release-notes source (secondary/search-summarized only).
- Did not check NumPy 2.0's whatsnew/migration guide for additional distribution-level stream-break entries beyond what's listed above.
- Did not confirm whether `distributions.c`'s `random_standard_normal` comment block cites Marsaglia & Tsang (2000) by name in-source — the fetch returned the constants/tables but not an explicit citation; treat as unverified rather than asserting a citation exists.

---

## scikit-learn: documented `random_state` / reproducibility policy

### Takeaway
scikit-learn's glossary and changelog conventions amount to an explicit acknowledgment that `random_state` reproducibility is a **within-version, within-install** guarantee only — not a cross-version guarantee. The user-guide "Controlling randomness" page covers reproducibility across repeated *program executions* within one install; the actual disclaimer that results can change across releases lives in the Glossary's "backwards compatibility" entry, which classifies prediction-changing bug fixes as an accepted exception, best-effort-flagged in changelogs, not guaranteed.

### Cited Findings
- **Glossary — `random_state`** (https://scikit-learn.org/stable/glossary.html#term-random_state), verbatim: "Controls the randomness of an estimator. Many estimators have a `random_state` parameter which can be set to a fixed integer seed to make their behaviour deterministic. In estimators that use randomness during fitting, if `random_state` is not set (or set to None), then each time `fit` is called, different results may be obtained. If an integer is provided, it seeds the internal random number generator before fitting. If an estimator's `random_state` parameter is set to an integer, an exact clone of the estimator will give the exact same results if fit on the same data. If `random_state` is set to a numpy.random.RandomState instance or a numpy.random.Generator instance, that instance will be used instead of creating a new seeded random number generator. See Controlling randomness for more details." This entry carries **no** cross-version caveat.
- **"Controlling randomness" user guide** (https://scikit-learn.org/stable/common_pitfalls.html#controlling-randomness), verbatim excerpts: "In order to obtain reproducible (i.e. constant) results across multiple program executions, we need to remove all uses of random_state=None, which is the default... We are now guaranteed that the result of this script will always be 0.84, no matter how many times we run it." "If an integer is passed, calling fit or split multiple times always yields the same results... If None or a RandomState instance is passed: fit and split will yield different results each time they are called." "We do not recommend setting the global numpy seed by calling np.random.seed(0)." This entire section is scoped to "within one version, one environment, repeated runs" — it says nothing about cross-version or cross-platform guarantees.
- **Glossary — "backwards compatibility" — the real version-stability disclaimer** (https://scikit-learn.org/stable/glossary.html#term-backwards-compatibility), verbatim: "We generally try to maintain backward compatibility (i.e. interfaces and behaviors may be extended but not changed or removed) from release to release but this comes with some exceptions: ... **Bug fixes and enhancements** Bug fixes and – less often – enhancements may change the behavior of estimators, **including the predictions of an estimator trained on the same data and random_state**. When this happens, we attempt to note it clearly in the changelog. ... Despite this informal contract with our users, the software is provided as is, as stated in the license. When a release inadvertently introduces changes that are not backward compatible, these are known as software regressions."
  - This is the closest scikit-learn comes to an explicit "no cross-version guarantee" statement — phrased as an accepted exception to an informal backward-compatibility contract, with only a best-effort changelog note, not a hard guarantee.
- **FAQ** (https://scikit-learn.org/stable/faq.html): no dedicated Q&A exists for "why do I get different results with the same random_state across versions"; the only related entry, "How do I set a random_state for an entire execution?", simply redirects to the Controlling randomness page.

### Inferences
- scikit-learn deliberately does **not** use words like "guarantee" or "frozen" for cross-version `random_state` stability (unlike NumPy's NEP 19 for `RandomState`) — its weaker, "attempt to note it in the changelog" framing is a meaningfully looser commitment than NumPy's.

### Gaps
- No scikit-learn page contains an explicit sentence of the exact form "we do not guarantee ... across versions" — the "backwards compatibility" glossary entry is the closest substantive equivalent but uses different wording; report this as "no such explicit sentence exists," not as a paraphrase presented as a quote.
- scikit-learn's own docs are silent on cross-platform (CPU architecture) determinism specifically (that topic is covered, for the underlying RNG, by NumPy's own compatibility policy instead).
- Not every scikit-learn page was checked (e.g. model_persistence.html) — an explicit cross-version disclaimer could conceivably exist elsewhere and was missed.

---

## Concrete documented examples of a version bump changing the random stream for a fixed seed

### Takeaway
NumPy is the only one of the three projects with a formal, named policy document governing stream compatibility, and its release notes give multiple concrete, explicitly-flagged instances of breaking changes (1.19.0: Dirichlet, SeedSequence+spawning, MT19937.jumped; 1.22.0: float32 uniform variates; 1.21.0: PCG64DXSM addition). scikit-learn's changelogs, while lacking a named policy page, do explicitly call out specific estimators whose `fit()` results changed for a fixed `random_state`. SciPy, notably, was **not** found to explicitly flag stream-breaking changes in its release notes in the same disclosed way — this is reported as a documented asymmetry/gap, not evidence that SciPy never breaks streams.

### Cited Findings
**NumPy** (see full verbatim quotes in the "NEP 19" and "distribution transforms" sections above): 1.19.0 (Dirichlet algorithm change, SeedSequence+spawn padding change, MT19937.jumped fix), 1.22.0 (float32 uniform LSB bug fix), 1.21.0 (PCG64DXSM addition, changing streams for parallel/spawned users only).

**scikit-learn — concrete changelog instances naming estimators**:
- v1.3.0 (https://scikit-learn.org/stable/whats_new/v1.3.html), verbatim: "The sample_weight parameter now will be used in centroids initialization for cluster.KMeans, cluster.BisectingKMeans and cluster.MiniBatchKMeans. **This change will break backward compatibility, since numbers generated from same random seeds will be different.**" (#25752)
- v0.22.0 (https://scikit-learn.org/stable/whats_new/v0.22.html), verbatim: "Fixed a bug where cluster.KMeans produced inconsistent results between n_jobs=1 and n_jobs>1 due to the handling of the random state." (#9288); and, same page, a fix that made a fixed seed *more* consistent than before: "inspection.permutation_importance will return the same importances when a random_state is given for both n_jobs=1 or n_jobs>1..." (#15933)

**SciPy**:
- No explicit "the stream will be different from previous releases"-style disclaimer was found in SciPy release notes checked (1.9.0 fetched directly; others searched more broadly).
- Related but not equivalent material found: (a) `scipy.stats.sampling`'s UNU.RAN-based generators (introduced SciPy 1.8.0) are documented as producing a different stream than the equivalent `scipy.stats` distribution's own `.rvs()` for the same seed — but this is a different-API-vs-different-API disclaimer, not a same-API-version-bump disclaimer; (b) SciPy 1.9.0 release notes mention speed-ups for `gennorm` and `nakagami` variate generation — an algorithmic rewrite that necessarily changes the output stream for a fixed seed, but the release notes do not explicitly say so.

### Inferences
- The SciPy asymmetry (no explicit disclosure convention, unlike NumPy and scikit-learn) is itself worth noting in the blog post as a cautionary example: the *absence* of a stated policy does not mean the stream is stable — it likely means SciPy breaks fixed-seed streams on algorithmic rewrites without a dedicated changelog callout, which is arguably a higher-risk situation for a blog reader relying on "no news is good news."

### Gaps
- A more exhaustive pass through every SciPy release-notes page (1.0–1.15+) was not performed given tool budget; this is the single highest-value follow-up if the final piece needs a SciPy example as strong as the NumPy/scikit-learn ones.
- The NumPy 1.16.4/1.17.0 "log-of-zero" stream-break claim (noted in the distribution-transforms section) was only secondary-sourced; needs independent primary-source verification before direct quotation.
- Did not check NumPy 2.0's full migration guide for additional stream-break entries beyond those already quoted.

---

## Overall gaps register (things explicitly NOT verified in this research pass)

- Full primary-source text of the 1998 Matsumoto & Nishimura TOMACS paper (ACM paywalled) — relied on abstract + reference code + secondary descriptions.
- Whether the 1998 paper itself specifies seeding at all (vs. treating it as later/out-of-scope).
- No official doc found stating side-by-side that CPython `random` and NumPy `RandomState` differ for the "same" seed — this is a sourced inference from comparing seeding code, not a quoted claim.
- O'Neill's PCG paper/pcg-random.org pages could not be fetched directly (repeated timeouts); PCG output-function variant names (XSH-RR etc.) are secondary-sourced only.
- O'Neill's own primary-source writeup of the DXSM design was not directly accessed (only referenced via a third-party blog that also failed to fetch).
- Exact NumPy default round count for Philox ("Philox4x64-10") is secondary-sourced, not confirmed on numpy.org directly.
- Full text/resolution of LWG issue 4134 regarding Philox specification ambiguity was not fetched.
- Whether `distributions.c`'s ziggurat implementation cites Marsaglia & Tsang (2000) by name in-source was not confirmed.
- The 1.16.4/1.17.0-era "log-of-zero" NumPy stream-break claim is secondary-sourced only.
- A comprehensive SciPy release-notes audit (beyond 1.9.0 and targeted searches) was not performed; no strong SciPy stream-break example was found, which is itself a notable gap/asymmetry versus NumPy and scikit-learn.
- Third-party claims about SeedSequence.spawn() thread-safety races (numpy/numpy#32063) and a doc-clarification issue (numpy/numpy#25778) were found only via secondary references, not fetched/confirmed against the primary tracker.
