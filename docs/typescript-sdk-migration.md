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
- LLMLingua-2 style compression now exists in the TypeScript core as a local-first stage

## Current Gaps Versus The Python Repo

- no direct importer for Python on-disk data yet
- no hosted API surface
- no connectors or document ingestion pipeline yet
- the default LLMLingua-2 ONNX checkpoint still needs to be staged in a Fraction-owned GitHub-hosted artifact path, though the export/validate/sync flow is now scriptable from `packages/fraction`

## Migration Path

1. Create a new Bun-side database with `Memory.open(...)` or `Fraction.open(...)`.
2. Re-add important memories through the SDK API.
3. Move application-level orchestration to either:
   - Promise API from `fraction`
   - Effect runtime API from `fraction/effect`
4. For model integrations, use:
   - `@fraction/vercel-ai`
   - `@fraction/tanstack-ai`
