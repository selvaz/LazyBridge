# Do I actually need the Advanced tier?

```mermaid
flowchart TD
    A[Do I need Advanced?] --> B{Are you writing…}
    B -->|a provider adapter| C[Yes - BaseProvider]
    B -->|a new execution strategy| D[Yes - Engine Protocol]
    B -->|a plan serialiser / worker| E[Yes - Plan.to_dict]
    B -->|app code using core.types| F[Probably not - reconsider]
    B -->|pipelines / prompts / models| G[No - use Basic/Mid/Full]
```

Advanced is framework authorship, not application development. Smell
test: if you're importing from `lazybridge.core.*` in app code, step
back — `from lazybridge import ...` covers 99% of use cases.
