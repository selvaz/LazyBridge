## question
Do I actually need the Advanced tier?

## tree
Adding support for a new LLM vendor?
    → BaseProvider + register_provider_rule    (Yes, Advanced.)

Replacing the execution loop with a non-LLM strategy (rules, RL, etc.)?
    → Engine Protocol                          (Yes, Advanced.)

Serialising a Plan to disk / over the wire?
    → Plan.to_dict / Plan.from_dict            (Yes, Advanced.)

Importing lazybridge.core.types directly in app code?
    → Usually a smell — you probably want Envelope / Agent / Tool.
      (STOP. Revisit Full tier.)

Tweaking prompts, swapping models, building pipelines?
    → Basic / Mid / Full cover this.           (STOP. You're fine.)

## tree_mermaid
flowchart TD
    A[Do I need Advanced?] --> B{Are you writing…}
    B -->|a provider adapter| C[Yes - BaseProvider]
    B -->|a new execution strategy| D[Yes - Engine Protocol]
    B -->|a plan serialiser / worker| E[Yes - Plan.to_dict]
    B -->|app code using core.types| F[Probably not - reconsider]
    B -->|pipelines / prompts / models| G[No - use Basic/Mid/Full]

## notes
Advanced is framework authorship, not application development. Smell
test: if you're importing from `lazybridge.core.*` in app code, step
back — `from lazybridge import ...` covers 99% of use cases.
