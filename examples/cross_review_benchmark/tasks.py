"""Model-visible specifications only. No hidden tests or solutions in this module."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    id: str
    category: str
    difficulty: str
    spec: str


def task(number, name, signature, doc, difficulty="medium", category="coding"):
    return Task(f"{category}_{number:02d}_{name}", category, difficulty, f'{signature}\n    """{doc}"""\n')


TASKS = [
    task(
        1,
        "gap_runs",
        "def gap_runs(values: list[int]) -> list[tuple[int, int]]:",
        "Return inclusive endpoints of maximal consecutive integer runs in sorted unique values. Ignore duplicates; accept negatives; empty yields []. Do not mutate values.",
        "easy",
    ),
    task(
        2,
        "clipped_windows",
        "def clipped_windows(text: str, width: int, stride: int) -> list[str]:",
        "For start positions 0, stride, ... strictly below len(text), return text[start:start+width], including short tails. Count Unicode code points. width and stride must be positive integers (bool is invalid), otherwise raise ValueError. Empty text returns [] after validation.",
        "easy",
    ),
    task(
        3,
        "cent_parts",
        "def cent_parts(amount: str, weights: list[int]) -> list[int]:",
        "Allocate nonnegative decimal amount in cents. Convert amount with Decimal and ROUND_HALF_UP to whole cents. Weights must be nonnegative integers (bool invalid), nonempty, with positive total; otherwise ValueError. Allocate floor(cents*w/total) then remaining cents to largest fractional remainders, ties by lower index. Nonfinite/negative amounts raise ValueError. Do not mutate weights.",
        "hard",
    ),
    task(
        4,
        "fold_counts",
        "def fold_counts(words: list[str]) -> list[tuple[str, int]]:",
        "Normalize each word with Unicode NFC, then casefold, then NFC again. Count normalized keys and return (key,count) in first occurrence order. Empty words are valid. Do not mutate words.",
        "medium",
    ),
    task(
        5,
        "utc_day",
        "def utc_day(stamp: str) -> str:",
        "Parse an ISO datetime using datetime.fromisoformat (accept trailing Z). Return its UTC date as YYYY-MM-DD. Naive timestamps and invalid timestamps raise ValueError. UTC conversion must occur before taking the date.",
        "medium",
    ),
    task(
        6,
        "month_anchor",
        "def month_anchor(day, offset: int):",
        "day is a datetime.date, offset is an integer month delta. Return a date offset months away, keeping the original day-of-month when possible, otherwise clipping to the destination month's last day. This is one direct shift, not repeated shifts. Year outside 1..9999 raises ValueError.",
        "hard",
    ),
    task(
        7,
        "quiet_ranges",
        "def quiet_ranges(busy: list[tuple[int, int]], start: int, stop: int) -> list[tuple[int, int]]:",
        "Return sorted maximal half-open intervals inside [start,stop) not covered by busy. Merge overlapping/touching busy ranges and clip them to the horizon. Reversed horizon or any reversed busy range raises ValueError; empty ranges ignored. Validate busy even for empty horizon. Do not mutate busy.",
        "hard",
    ),
    task(
        8,
        "escaped_fields",
        "def escaped_fields(text: str) -> list[str]:",
        r"Split at unescaped |. A backslash escapes exactly the following character, regardless of what that character is; remove the escaping backslash. A trailing unmatched backslash raises ValueError. Preserve empty fields; empty text returns [''].",
        "medium",
    ),
    task(
        9,
        "carry_round",
        "def carry_round(values: list[str]) -> list[int]:",
        "Parse finite Decimal strings (nonfinite raises ValueError). For each prefix sum, round to integer with ROUND_HALF_EVEN. Return differences between successive rounded prefix sums, with initial rounded total zero. Negative values allowed. Do not round each item independently; do not mutate input.",
        "hard",
    ),
    task(
        10,
        "recent_keys",
        "def recent_keys(keys: list[str], capacity: int) -> list[str]:",
        "Simulate most-recently-used cache keys. A hit moves key to newest; miss appends; evict oldest beyond capacity. Return surviving keys oldest to newest. capacity must be a nonnegative integer (bool invalid), otherwise ValueError. Zero capacity returns []. Do not mutate keys.",
        "easy",
    ),
    task(
        11,
        "braid_rows",
        "def braid_rows(rows: list[list[int]]) -> list[int]:",
        "Visit column 0 across rows top-to-bottom, column 1 bottom-to-top, alternating thereafter. Skip missing ragged cells. Empty rows allowed. Return all values in this order without mutating rows.",
        "medium",
    ),
    task(
        12,
        "stable_scores",
        "def stable_scores(rows: list[tuple[str, float]]) -> list[str]:",
        "Return names sorted by descending numeric score, preserving input order on ties. NaN scores come after every non-NaN score, preserving their order. Infinities are valid, signed zeros tie. Do not mutate rows.",
        "medium",
    ),
    task(
        13,
        "near_duplicate",
        "def near_duplicate(values: list[int], distance: int) -> bool:",
        "True iff equal values occur at two distinct indices with index difference <= distance. Negative distance raises ValueError. Zero distance always false. Do not mutate values.",
        "easy",
    ),
    task(
        14,
        "path_reduce",
        "def path_reduce(path: str) -> str:",
        "Lexically normalize a slash-separated POSIX-style path without filesystem access. Drop empty components and '.', resolve '..' against preceding ordinary component. Preserve unmatched '..' in relative paths, discard them at absolute root. Empty relative result is '.', empty absolute is '/'. Backslash is an ordinary character. Discard trailing slash.",
        "medium",
    ),
    task(
        15,
        "select_rank",
        "def select_rank(values: list[int], rank: int) -> int:",
        "Return the element at zero-based rank in ascending order, counting duplicates. Invalid rank (<0 or >=len(values)) raises IndexError. Do not mutate values.",
        "easy",
    ),
    task(
        16,
        "dense_labels",
        "def dense_labels(labels: list[str]) -> list[int]:",
        "Assign first-seen distinct strings IDs 0,1,..., return each label's ID. Compare strings exactly (no normalization). Empty list returns []. Do not mutate labels.",
        "easy",
    ),
    task(
        17,
        "retry_slots",
        "def retry_slots(delays: list[int], budget: int) -> list[int]:",
        "Return cumulative scheduled retry times starting at time zero (do not include initial zero separately). Include each next retry iff its cumulative time <= budget; stop at first over-budget retry. Zero delays allowed. Negative budget or any negative delay raises ValueError, including delays after an over-budget prefix. Do not mutate delays.",
        "medium",
    ),
    task(
        18,
        "end_match",
        "def end_match(text: str, suffixes: list[str]) -> str | None:",
        "Among suffixes for which text.endswith(suffix), return the longest by Unicode code point length, ties by first occurrence. Empty suffix matches every text. Return None if none match. Case-sensitive; do not mutate suffixes.",
        "easy",
    ),
    task(
        19,
        "local_midnights",
        "def local_midnights(start, stop) -> list:",
        "start and stop are timezone-aware datetimes using the same tzinfo object. Return aware datetimes at local midnight whose instants are in [start,stop), in chronological order. Enumerate calendar dates in their shared zone; compare instants in UTC (including folded times). Raise ValueError for naive inputs, different tzinfo objects, or reversed instant range. Include start if midnight, exclude stop. Use Python ZoneInfo semantics when constructing midnight.",
        "hard",
    ),
    task(
        20,
        "aligned_batches",
        "def aligned_batches(items: list[int], size: int, offset: int) -> list[list[int]]:",
        "Partition items in order at global index multiples of size: item i has global index offset+i. The first batch may be short if offset is unaligned (including negative offset). No empty batches; empty items returns []. size must be a positive integer (bool invalid), otherwise ValueError. Return fresh lists without mutating items.",
        "hard",
    ),
    task(
        1,
        "tool_name",
        "def make_tool(fn):",
        "Use the real LazyBridge public API to wrap Python callable fn as a tool explicitly named 'measure' with description 'Measure a value'. Return the Tool. No model calls. Use documented parameters; no substitute objects.",
        "easy",
        "api",
    ),
    task(
        2,
        "mock_plan",
        "def run_mock_plan(text: str) -> str:",
        "Use real LazyBridge Agent, Plan, Step and MockAgent to create a one-step deterministic pipeline. A MockAgent named 'echo' returns text. The pipeline must compose the subagent in tools and run the step by its name. Return the resulting envelope's text. No real provider calls.",
        "medium",
        "api",
    ),
    task(
        3,
        "envelope_context",
        "def make_envelope(task: str, context: str):",
        "Use the real LazyBridge public API to construct and return an Envelope with task and context set to the supplied values, and payload equal to {'status': 'queued'}. No provider calls.",
        "easy",
        "api",
    ),
    task(
        4,
        "review_agent",
        "def make_review_agent(engine, judge):",
        "Use the real LazyBridge public API to construct an Agent named 'checked' using the supplied engine and judge for output verification with a maximum of two verification attempts. Return the Agent without running it. No provider calls.",
        "medium",
        "api",
    ),
    task(
        5,
        "step_context",
        "def make_context_step():",
        "Use real LazyBridge public API to return a Step targeting 'writer', task from previous step output, context from the step named 'research'. Use the public sentinel helpers. Do not construct a pipeline or run a provider.",
        "medium",
        "api",
    ),
]
