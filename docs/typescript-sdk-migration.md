# Fraction TypeScript SDK Migration Notes

## Summary

The Bun SDK is a native TypeScript implementation. It does not wrap the Python runtime and it does not reuse the Python persistence format.

## What Carries Over

- local-first storage
- hybrid retrieval
- memory versioning
- soft-forget semantics
- history
- scope-aware recall

## What Changed

- persistence is SQLite-backed instead of JSON sidecars
- the implementation model is Effect v4
- the primary Bun entrypoints are `fraction` and `fraction/effect`
- Vercel AI SDK and TanStack AI integrations are separate adapter packages

## Current Gaps Versus The Python Repo

- no direct importer for Python on-disk data yet
- no hosted API surface
- no connectors or document ingestion pipeline yet
- local embedding path is currently lightweight and deterministic rather than model-backed

## Migration Path

1. Create a new Bun-side database with `Memory.open(...)` or `Fraction.open(...)`.
2. Re-add important memories through the SDK API.
3. Move application-level orchestration to either:
   - Promise API from `fraction`
   - Effect runtime API from `fraction/effect`
4. For model integrations, use:
   - `@fraction/vercel-ai`
   - `@fraction/tanstack-ai`
