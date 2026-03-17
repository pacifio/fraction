import { SqliteClient } from "@effect/sql-sqlite-bun"
import {
  Effect,
  Exit,
  Layer,
  ManagedRuntime,
  Request,
  RequestResolver,
  Schema,
  ServiceMap
} from "effect"
import * as SqlClient from "effect/unstable/sql/SqlClient"

import {
  EmbeddingError,
  ExtractionError,
  InvalidScope,
  MemoryNotFound,
  MigrationError,
  RetrievalError,
  StorageError
} from "./errors"
import { DEFAULT_CONFIG, FractionConfigSchema } from "./types"
import type {
  ExtractionResult,
  FilterExpr,
  FractionConfig,
  FractionConfigInput,
  HistoryEvent,
  JsonRecord,
  MemoryRecord,
  RecallOptions,
  Scope,
  SearchResult
} from "./types"
import {
  compressText,
  cosineSimilarity,
  decodeVector,
  embedTextLocal,
  encodeVector,
  extractEntities,
  extractEventAt,
  hashText,
  hasScope,
  jsonToMetadata,
  jsonToScope,
  makeMemoryId,
  matchesFilter,
  messagesToText,
  normalizeScope,
  nowIso,
  rrfScore,
  scopeToJson
} from "./utils"

export class FractionConfigRef extends ServiceMap.Service<FractionConfigRef, FractionConfig>()(
  "fraction/FractionConfig"
) {}

export class MigrationService extends ServiceMap.Service<
  MigrationService,
  {
    readonly run: Effect.Effect<void, MigrationError>
  }
>()("fraction/MigrationService") {
  static readonly layer = Layer.effect(
    MigrationService,
    Effect.gen(function* () {
      const sql = yield* SqlClient.SqlClient
      return MigrationService.of({
        run: sql
          .withTransaction(
            Effect.gen(function* () {
              yield* sql`CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            root_id TEXT NOT NULL,
            parent_id TEXT,
            version INTEGER NOT NULL,
            content TEXT NOT NULL,
            raw_content TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            scope_json TEXT NOT NULL,
            namespace TEXT,
            user_id TEXT,
            agent_id TEXT,
            run_id TEXT,
            app_id TEXT,
            content_hash TEXT NOT NULL,
            event_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            forgotten_at TEXT,
            is_latest INTEGER NOT NULL DEFAULT 1
          )`
              yield* sql`CREATE INDEX IF NOT EXISTS memories_scope_idx ON memories(namespace, user_id, agent_id, run_id, app_id, is_latest)`
              yield* sql`CREATE INDEX IF NOT EXISTS memories_root_idx ON memories(root_id, version)`
              yield* sql`CREATE TABLE IF NOT EXISTS memory_versions (
            id TEXT PRIMARY KEY,
            root_id TEXT NOT NULL,
            parent_id TEXT,
            memory_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
          )`
              yield* sql`CREATE TABLE IF NOT EXISTS memory_history (
            id TEXT PRIMARY KEY,
            root_id TEXT NOT NULL,
            memory_id TEXT NOT NULL,
            event TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
          )`
              yield* sql`CREATE TABLE IF NOT EXISTS memory_embeddings (
            memory_id TEXT PRIMARY KEY,
            vector BLOB NOT NULL,
            dimensions INTEGER NOT NULL
          )`
              yield* sql`CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
            id UNINDEXED,
            content,
            raw_content
          )`
              yield* sql`CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            normalized TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL
          )`
              yield* sql`CREATE TABLE IF NOT EXISTS memory_entities (
            memory_id TEXT NOT NULL,
            entity_id INTEGER NOT NULL,
            PRIMARY KEY(memory_id, entity_id)
          )`
              yield* sql`CREATE TABLE IF NOT EXISTS entity_edges (
            source_entity_id INTEGER NOT NULL,
            target_entity_id INTEGER NOT NULL,
            weight INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY(source_entity_id, target_entity_id)
          )`
            })
          )
          .pipe(
            Effect.mapError(
              (cause) => new MigrationError({ message: "Failed to run SQLite migrations", cause })
            )
          )
      })
    })
  )
}

type MemoryRow = {
  id: string
  root_id: string
  parent_id: string | null
  version: number
  content: string
  raw_content: string
  metadata_json: string
  scope_json: string
  created_at: string
  updated_at: string
  event_at: string | null
  forgotten_at: string | null
  is_latest: number
}

type HistoryRow = {
  id: string
  root_id: string
  memory_id: string
  event: HistoryEvent["event"]
  content: string
  created_at: string
}

const toOptionalFields = (fields: {
  readonly parentId?: string | null | undefined
  readonly eventAt?: string | null | undefined
  readonly forgottenAt?: string | null | undefined
}) =>
  ({
    ...(fields.parentId ? { parentId: fields.parentId } : {}),
    ...(fields.eventAt ? { eventAt: fields.eventAt } : {}),
    ...(fields.forgottenAt ? { forgottenAt: fields.forgottenAt } : {})
  }) satisfies Partial<Pick<MemoryRecord, "parentId" | "eventAt" | "forgottenAt">>

const createMemoryRecord = (input: {
  readonly id: MemoryRecord["id"]
  readonly rootId: string
  readonly version: number
  readonly content: string
  readonly rawContent: string
  readonly metadata: JsonRecord
  readonly scope: Scope
  readonly createdAt: string
  readonly updatedAt: string
  readonly isLatest: boolean
  readonly parentId?: string | null | undefined
  readonly eventAt?: string | null | undefined
  readonly forgottenAt?: string | null | undefined
}): MemoryRecord => ({
  id: input.id,
  rootId: input.rootId,
  version: input.version,
  content: input.content,
  rawContent: input.rawContent,
  metadata: input.metadata,
  scope: input.scope,
  createdAt: input.createdAt,
  updatedAt: input.updatedAt,
  isLatest: input.isLatest,
  ...toOptionalFields(input)
})

const rowToRecord = (row: MemoryRow): MemoryRecord =>
  createMemoryRecord({
    id: row.id as MemoryRecord["id"],
    rootId: row.root_id,
    parentId: row.parent_id,
    version: row.version,
    content: row.content,
    rawContent: row.raw_content,
    metadata: jsonToMetadata(row.metadata_json),
    scope: jsonToScope(row.scope_json),
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    eventAt: row.event_at,
    forgottenAt: row.forgotten_at,
    isLatest: row.is_latest === 1
  })

const historyRowToEvent = (row: HistoryRow): HistoryEvent => ({
  id: row.id,
  rootId: row.root_id,
  memoryId: row.memory_id,
  event: row.event,
  content: row.content,
  createdAt: row.created_at
})

const applyScopeClause = (scope: Scope) =>
  [
    "namespace = ? AND COALESCE(user_id, '') = ? AND COALESCE(agent_id, '') = ? AND COALESCE(run_id, '') = ? AND COALESCE(app_id, '') = ?",
    [
      scope.namespace ?? "",
      scope.userId ?? "",
      scope.agentId ?? "",
      scope.runId ?? "",
      scope.appId ?? ""
    ]
  ] as const

export class MemoryRepo extends ServiceMap.Service<
  MemoryRepo,
  {
    readonly insert: (
      record: MemoryRecord,
      embedding: Float32Array,
      entities: ReadonlyArray<string>
    ) => Effect.Effect<void, StorageError>
    readonly updateVersion: (
      previous: MemoryRecord,
      next: MemoryRecord,
      embedding: Float32Array,
      entities: ReadonlyArray<string>
    ) => Effect.Effect<void, StorageError>
    readonly listActive: (scope: Scope) => Effect.Effect<ReadonlyArray<MemoryRecord>, StorageError>
    readonly listAll: (scope: Scope) => Effect.Effect<ReadonlyArray<MemoryRecord>, StorageError>
    readonly getById: (id: string) => Effect.Effect<MemoryRecord | null, StorageError>
    readonly forget: (id: string, forgottenAt: string) => Effect.Effect<void, StorageError>
    readonly deleteRoot: (id: string) => Effect.Effect<void, StorageError>
    readonly deleteScope: (scope: Scope) => Effect.Effect<void, StorageError>
    readonly reset: Effect.Effect<void, StorageError>
    readonly searchLexical: (
      query: string,
      scope: Scope,
      limit: number
    ) => Effect.Effect<ReadonlyArray<{ readonly id: string; readonly score: number }>, StorageError>
    readonly embeddings: (
      scope: Scope
    ) => Effect.Effect<
      ReadonlyArray<{ readonly id: string; readonly vector: Float32Array }>,
      StorageError
    >
    readonly relatedByEntities: (
      queryEntities: ReadonlyArray<string>,
      scope: Scope
    ) => Effect.Effect<ReadonlyArray<{ readonly id: string; readonly score: number }>, StorageError>
    readonly history: (memoryId: string) => Effect.Effect<ReadonlyArray<HistoryEvent>, StorageError>
    readonly recordHistory: (event: HistoryEvent) => Effect.Effect<void, StorageError>
  }
>()("fraction/MemoryRepo") {
  static readonly layer = Layer.effect(
    MemoryRepo,
    Effect.gen(function* () {
      const sql = yield* SqlClient.SqlClient

      const ensureEntityIds = (entities: ReadonlyArray<string>) =>
        Effect.forEach(
          entities,
          (entity) =>
            Effect.gen(function* () {
              const normalized = entity.toLowerCase()
              yield* sql.unsafe("INSERT OR IGNORE INTO entities (normalized, name) VALUES (?, ?)", [
                normalized,
                entity
              ])
              const rows = yield* sql.unsafe<{ id: number }>(
                "SELECT id FROM entities WHERE normalized = ?",
                [normalized]
              )
              return rows[0]?.id ?? 0
            }),
          { concurrency: 1 }
        )

      const storeEntities = (memoryId: string, entities: ReadonlyArray<string>) =>
        Effect.gen(function* () {
          const ids = (yield* ensureEntityIds(entities)).filter((id) => id > 0)
          for (const entityId of ids) {
            yield* sql.unsafe(
              "INSERT OR IGNORE INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
              [memoryId, entityId]
            )
          }
          for (let left = 0; left < ids.length; left++) {
            for (let right = left + 1; right < ids.length; right++) {
              const source = ids[left]!
              const target = ids[right]!
              yield* sql.unsafe(
                `INSERT INTO entity_edges (source_entity_id, target_entity_id, weight)
                 VALUES (?, ?, 1)
                 ON CONFLICT(source_entity_id, target_entity_id)
                 DO UPDATE SET weight = weight + 1`,
                [source, target]
              )
              yield* sql.unsafe(
                `INSERT INTO entity_edges (source_entity_id, target_entity_id, weight)
                 VALUES (?, ?, 1)
                 ON CONFLICT(source_entity_id, target_entity_id)
                 DO UPDATE SET weight = weight + 1`,
                [target, source]
              )
            }
          }
        })

      const insert = (
        record: MemoryRecord,
        embedding: Float32Array,
        entities: ReadonlyArray<string>
      ) =>
        sql
          .withTransaction(
            Effect.gen(function* () {
              yield* sql.unsafe(
                `INSERT INTO memories
            (id, root_id, parent_id, version, content, raw_content, metadata_json, scope_json, namespace, user_id, agent_id, run_id, app_id, content_hash, event_at, created_at, updated_at, forgotten_at, is_latest)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
                [
                  record.id,
                  record.rootId,
                  record.parentId ?? null,
                  record.version,
                  record.content,
                  record.rawContent,
                  JSON.stringify(record.metadata),
                  scopeToJson(record.scope),
                  record.scope.namespace ?? null,
                  record.scope.userId ?? null,
                  record.scope.agentId ?? null,
                  record.scope.runId ?? null,
                  record.scope.appId ?? null,
                  hashText(record.content),
                  record.eventAt ?? null,
                  record.createdAt,
                  record.updatedAt,
                  record.forgottenAt ?? null,
                  record.isLatest ? 1 : 0
                ]
              )
              yield* sql.unsafe(
                `INSERT INTO memory_versions (id, root_id, parent_id, memory_id, version, content, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?)`,
                [
                  makeMemoryId(),
                  record.rootId,
                  record.parentId ?? null,
                  record.id,
                  record.version,
                  record.content,
                  record.createdAt
                ]
              )
              yield* sql.unsafe(
                `INSERT INTO memory_embeddings (memory_id, vector, dimensions) VALUES (?, ?, ?)`,
                [record.id, encodeVector(embedding), embedding.length]
              )
              yield* sql.unsafe(
                `INSERT INTO memory_fts (id, content, raw_content) VALUES (?, ?, ?)`,
                [record.id, record.content, record.rawContent]
              )
              yield* storeEntities(record.id, entities)
            })
          )
          .pipe(
            Effect.mapError(
              (cause) => new StorageError({ message: "Failed to insert memory", cause })
            )
          )

      const updateVersion = (
        previous: MemoryRecord,
        next: MemoryRecord,
        embedding: Float32Array,
        entities: ReadonlyArray<string>
      ) =>
        sql
          .withTransaction(
            Effect.gen(function* () {
              yield* sql.unsafe("UPDATE memories SET is_latest = 0, updated_at = ? WHERE id = ?", [
                next.updatedAt,
                previous.id
              ])
              yield* insert(next, embedding, entities)
            })
          )
          .pipe(
            Effect.mapError(
              (cause) => new StorageError({ message: "Failed to update memory version", cause })
            )
          )

      const listActive = (scope: Scope) => {
        const [clause, params] = applyScopeClause(scope)
        return sql
          .unsafe<MemoryRow>(
            `SELECT * FROM memories WHERE ${clause} AND is_latest = 1 AND forgotten_at IS NULL ORDER BY updated_at DESC`,
            params
          )
          .pipe(
            Effect.map((rows) => rows.map(rowToRecord)),
            Effect.mapError(
              (cause) => new StorageError({ message: "Failed to list active memories", cause })
            )
          )
      }

      const listAll = (scope: Scope) => {
        const [clause, params] = applyScopeClause(scope)
        return sql
          .unsafe<MemoryRow>(
            `SELECT * FROM memories WHERE ${clause} ORDER BY updated_at DESC`,
            params
          )
          .pipe(
            Effect.map((rows) => rows.map(rowToRecord)),
            Effect.mapError(
              (cause) => new StorageError({ message: "Failed to list memories", cause })
            )
          )
      }

      const getById = (id: string) =>
        sql.unsafe<MemoryRow>("SELECT * FROM memories WHERE id = ? LIMIT 1", [id]).pipe(
          Effect.map((rows) => (rows[0] ? rowToRecord(rows[0]) : null)),
          Effect.mapError((cause) => new StorageError({ message: "Failed to get memory", cause }))
        )

      const forget = (id: string, forgottenAt: string) =>
        sql
          .unsafe("UPDATE memories SET forgotten_at = ?, updated_at = ? WHERE id = ?", [
            forgottenAt,
            forgottenAt,
            id
          ])
          .pipe(
            Effect.asVoid,
            Effect.mapError(
              (cause) => new StorageError({ message: "Failed to forget memory", cause })
            )
          )

      const deleteRoot = (id: string) =>
        Effect.gen(function* () {
          const record = yield* getById(id)
          if (!record) {
            return
          }
          const rows = yield* sql.unsafe<{ id: string }>(
            "SELECT id FROM memories WHERE root_id = ?",
            [record.rootId]
          )
          const ids = rows.map((row) => row.id)
          if (ids.length === 0) {
            return
          }
          const placeholders = ids.map(() => "?").join(",")
          yield* sql.withTransaction(
            Effect.gen(function* () {
              yield* sql.unsafe(
                `DELETE FROM memory_embeddings WHERE memory_id IN (${placeholders})`,
                ids
              )
              yield* sql.unsafe(`DELETE FROM memory_fts WHERE id IN (${placeholders})`, ids)
              yield* sql.unsafe(
                `DELETE FROM memory_entities WHERE memory_id IN (${placeholders})`,
                ids
              )
              yield* sql.unsafe(`DELETE FROM memories WHERE id IN (${placeholders})`, ids)
            })
          )
        }).pipe(
          Effect.mapError(
            (cause) => new StorageError({ message: "Failed to delete memory chain", cause })
          )
        )

      const deleteByIds = (ids: ReadonlyArray<string>) => {
        if (ids.length === 0) {
          return Effect.void
        }
        const placeholders = ids.map(() => "?").join(",")
        return sql.withTransaction(
          Effect.gen(function* () {
            yield* sql.unsafe(
              `DELETE FROM memory_embeddings WHERE memory_id IN (${placeholders})`,
              ids
            )
            yield* sql.unsafe(`DELETE FROM memory_fts WHERE id IN (${placeholders})`, ids)
            yield* sql.unsafe(
              `DELETE FROM memory_entities WHERE memory_id IN (${placeholders})`,
              ids
            )
            yield* sql.unsafe(
              `DELETE FROM memory_versions WHERE memory_id IN (${placeholders})`,
              ids
            )
            yield* sql.unsafe(
              `DELETE FROM memory_history WHERE memory_id IN (${placeholders})`,
              ids
            )
            yield* sql.unsafe(`DELETE FROM memories WHERE id IN (${placeholders})`, ids)
          })
        )
      }

      const deleteScope = (scope: Scope) => {
        const [clause, params] = applyScopeClause(scope)
        return Effect.gen(function* () {
          const rows = yield* sql.unsafe<{ id: string }>(
            `SELECT id FROM memories WHERE ${clause}`,
            params
          )
          const ids = rows.map((row) => row.id)
          yield* deleteByIds(ids)
        }).pipe(
          Effect.mapError(
            (cause) => new StorageError({ message: "Failed to delete memories for scope", cause })
          )
        )
      }

      const reset = sql
        .withTransaction(
          Effect.gen(function* () {
            yield* sql.unsafe("DELETE FROM memory_embeddings")
            yield* sql.unsafe("DELETE FROM memory_fts")
            yield* sql.unsafe("DELETE FROM memory_entities")
            yield* sql.unsafe("DELETE FROM entity_edges")
            yield* sql.unsafe("DELETE FROM entities")
            yield* sql.unsafe("DELETE FROM memory_versions")
            yield* sql.unsafe("DELETE FROM memory_history")
            yield* sql.unsafe("DELETE FROM memories")
          })
        )
        .pipe(
          Effect.mapError(
            (cause) => new StorageError({ message: "Failed to reset database", cause })
          )
        )

      const searchLexical = (query: string, scope: Scope, limit: number) => {
        const [clause, params] = applyScopeClause(scope)
        const tokens = query.match(/[A-Za-z0-9_]+/g) ?? []
        if (tokens.length === 0) {
          return Effect.succeed(
            [] as ReadonlyArray<{ readonly id: string; readonly score: number }>
          )
        }
        const q = tokens.map((token) => `"${token}"`).join(" OR ")
        return sql
          .unsafe<{ id: string; score: number }>(
            `SELECT fts.id AS id, bm25(memory_fts) AS score
           FROM memory_fts AS fts
           JOIN memories AS memories ON memories.id = fts.id
           WHERE memory_fts MATCH ? AND ${clause} AND memories.is_latest = 1 AND memories.forgotten_at IS NULL
           ORDER BY bm25(memory_fts) ASC
           LIMIT ?`,
            [q, ...params, limit]
          )
          .pipe(
            Effect.map((rows) =>
              rows.map((row, index) => ({
                id: row.id,
                score: 1 / (1 + index + Math.abs(row.score))
              }))
            ),
            Effect.mapError(
              (cause) => new StorageError({ message: "Failed lexical search", cause })
            )
          )
      }

      const embeddings = (scope: Scope) => {
        const [clause, params] = applyScopeClause(scope)
        return sql
          .unsafe<{ id: string; vector: Uint8Array }>(
            `SELECT memories.id AS id, memory_embeddings.vector AS vector
           FROM memories
           JOIN memory_embeddings ON memory_embeddings.memory_id = memories.id
           WHERE ${clause} AND memories.is_latest = 1 AND memories.forgotten_at IS NULL`,
            params
          )
          .pipe(
            Effect.map((rows) =>
              rows.map((row) => ({ id: row.id, vector: decodeVector(row.vector) }))
            ),
            Effect.mapError(
              (cause) => new StorageError({ message: "Failed to load embeddings", cause })
            )
          )
      }

      const relatedByEntities = (queryEntities: ReadonlyArray<string>, scope: Scope) => {
        if (queryEntities.length === 0) {
          return Effect.succeed(
            [] as ReadonlyArray<{ readonly id: string; readonly score: number }>
          )
        }
        const normalized = queryEntities.map((entity) => entity.toLowerCase())
        const placeholders = normalized.map(() => "?").join(",")
        const [clause, params] = applyScopeClause(scope)
        return sql
          .unsafe<{ id: string; score: number }>(
            `SELECT DISTINCT memories.id AS id, COALESCE(SUM(entity_edges.weight), 1) AS score
           FROM memories
           JOIN memory_entities ON memory_entities.memory_id = memories.id
           JOIN entities ON entities.id = memory_entities.entity_id
           LEFT JOIN entity_edges ON entity_edges.source_entity_id = entities.id
           WHERE entities.normalized IN (${placeholders}) AND ${clause} AND memories.is_latest = 1 AND memories.forgotten_at IS NULL
           GROUP BY memories.id
           ORDER BY score DESC`,
            [...normalized, ...params]
          )
          .pipe(
            Effect.mapError((cause) => new StorageError({ message: "Failed graph search", cause }))
          )
      }

      const history = (memoryId: string) =>
        Effect.gen(function* () {
          const record = yield* getById(memoryId)
          if (!record) {
            return [] as ReadonlyArray<HistoryEvent>
          }
          const rows = yield* sql.unsafe<HistoryRow>(
            "SELECT * FROM memory_history WHERE root_id = ? ORDER BY created_at ASC",
            [record.rootId]
          )
          return rows.map(historyRowToEvent)
        }).pipe(
          Effect.mapError((cause) => new StorageError({ message: "Failed to read history", cause }))
        )

      const recordHistory = (event: HistoryEvent) =>
        sql
          .unsafe(
            `INSERT INTO memory_history (id, root_id, memory_id, event, content, created_at)
           VALUES (?, ?, ?, ?, ?, ?)`,
            [event.id, event.rootId, event.memoryId, event.event, event.content, event.createdAt]
          )
          .pipe(
            Effect.asVoid,
            Effect.mapError(
              (cause) => new StorageError({ message: "Failed to record history", cause })
            )
          )

      return MemoryRepo.of({
        insert,
        updateVersion,
        listActive,
        listAll,
        getById,
        forget,
        deleteRoot,
        deleteScope,
        reset,
        searchLexical,
        embeddings,
        relatedByEntities,
        history,
        recordHistory
      })
    })
  )
}

interface EmbedTextRequest extends Request.Request<Float32Array, EmbeddingError> {
  readonly _tag: "fraction/EmbedTextRequest"
  readonly text: string
}

const EmbedTextRequest = Request.tagged<EmbedTextRequest>("fraction/EmbedTextRequest")

export class EmbeddingService extends ServiceMap.Service<
  EmbeddingService,
  {
    readonly embed: (text: string) => Effect.Effect<Float32Array, EmbeddingError>
    readonly embedMany: (
      texts: ReadonlyArray<string>
    ) => Effect.Effect<ReadonlyArray<Float32Array>, EmbeddingError>
  }
>()("fraction/EmbeddingService") {
  static readonly layer = Layer.effect(
    EmbeddingService,
    Effect.gen(function* () {
      const config = yield* FractionConfigRef
      const provider = config.embeddingProvider ?? {
        embed: (text: string) => embedTextLocal(text, config.embeddingDimensions),
        embedMany: (texts: ReadonlyArray<string>) =>
          texts.map((text) => embedTextLocal(text, config.embeddingDimensions))
      }

      const runBatch = (texts: ReadonlyArray<string>) =>
        Effect.tryPromise({
          try: async () => {
            const output = provider.embedMany
              ? await provider.embedMany(texts)
              : await Promise.all(texts.map((text) => provider.embed(text)))

            if (output.length !== texts.length) {
              throw new TypeError(
                `Expected ${texts.length} embeddings but received ${output.length}`
              )
            }

            return output.map((embedding) =>
              embedding instanceof Float32Array ? embedding : Float32Array.from(embedding)
            )
          },
          catch: (cause) => new EmbeddingError({ message: "Failed to embed text", cause })
        })

      const resolver = RequestResolver.make<EmbedTextRequest>((entries) =>
        runBatch(entries.map((entry) => entry.request.text)).pipe(
          Effect.matchEffect({
            onFailure: (error) =>
              Effect.sync(() => {
                for (const entry of entries) {
                  entry.completeUnsafe(Exit.fail(error))
                }
              }),
            onSuccess: (embeddings) =>
              Effect.sync(() => {
                entries.forEach((entry, index) => {
                  entry.completeUnsafe(Exit.succeed(embeddings[index]!))
                })
              })
          })
        )
      )

      const embed = (text: string) => Effect.request(EmbedTextRequest({ text }), resolver)
      const embedMany = (texts: ReadonlyArray<string>) =>
        texts.length === 0 ? Effect.succeed([] as ReadonlyArray<Float32Array>) : runBatch(texts)

      return EmbeddingService.of({ embed, embedMany })
    })
  )
}

export class ExtractionService extends ServiceMap.Service<
  ExtractionService,
  {
    readonly compress: (text: string) => Effect.Effect<ExtractionResult, ExtractionError>
  }
>()("fraction/ExtractionService") {
  static readonly layer = Layer.effect(
    ExtractionService,
    Effect.gen(function* () {
      const config = yield* FractionConfigRef
      const compress = (text: string) => {
        if (config.extractionProvider) {
          return Effect.tryPromise({
            try: async () => {
              const extracted = await config.extractionProvider!.extract(text)
              const content =
                extracted.content.trim().length > 0
                  ? extracted.content.trim()
                  : compressText(text, config.maxFactsPerInput)
              const entities =
                extracted.entities.length > 0 ? extracted.entities : extractEntities(content)
              return {
                content,
                entities: [...new Set(entities.map((entity) => entity.trim()).filter(Boolean))],
                ...(extracted.eventAt ? { eventAt: extracted.eventAt } : {})
              } satisfies ExtractionResult
            },
            catch: (cause) =>
              new ExtractionError({ message: "Failed to extract memory facts", cause })
          })
        }

        return Effect.sync(() => {
          const content = compressText(text, config.maxFactsPerInput)
          const eventAt = extractEventAt(text)
          return {
            content,
            entities: extractEntities(content),
            ...(eventAt ? { eventAt } : {})
          } satisfies ExtractionResult
        }).pipe(
          Effect.mapError(
            (cause) => new ExtractionError({ message: "Failed to extract memory facts", cause })
          )
        )
      }
      return ExtractionService.of({ compress })
    })
  )
}

export class RetrievalService extends ServiceMap.Service<
  RetrievalService,
  {
    readonly search: (
      query: string,
      scope: Scope,
      filter: FilterExpr | undefined,
      limit: number
    ) => Effect.Effect<ReadonlyArray<SearchResult>, RetrievalError | StorageError>
  }
>()("fraction/RetrievalService") {
  static readonly layer = Layer.effect(
    RetrievalService,
    Effect.gen(function* () {
      const config = yield* FractionConfigRef
      const repo = yield* MemoryRepo
      const embeddings = yield* EmbeddingService
      const search = (query: string, scope: Scope, filter: FilterExpr | undefined, limit: number) =>
        Effect.gen(function* () {
          const active = yield* repo.listActive(scope)
          const filtered = active.filter((memory) =>
            matchesFilter(memory.metadata as JsonRecord, filter)
          )
          const queryVector = yield* embeddings.embed(query)
          const vectorRanking = (yield* repo.embeddings(scope))
            .map((entry) => ({ id: entry.id, score: cosineSimilarity(queryVector, entry.vector) }))
            .sort((left, right) => right.score - left.score)
            .slice(0, limit * 3)
          const lexicalRanking = yield* repo.searchLexical(query, scope, limit * 3)
          const graphRanking = yield* repo.relatedByEntities(extractEntities(query), scope)
          const queryTime = extractEventAt(query)
          const temporalRanking = queryTime
            ? filtered
                .map((memory) => {
                  const memoryTime = memory.eventAt ?? memory.createdAt
                  const diff = Math.abs(
                    new Date(queryTime).getTime() - new Date(memoryTime).getTime()
                  )
                  return { id: memory.id, score: 1 / (1 + diff / 86_400_000) }
                })
                .sort((left, right) => right.score - left.score)
                .slice(0, limit * 3)
            : []
          const fused = rrfScore(
            [lexicalRanking, vectorRanking, graphRanking, temporalRanking],
            [config.lexicalWeight, config.vectorWeight, config.graphWeight, config.temporalWeight],
            config.rrfK
          )
          return filtered
            .map((memory) => {
              const hit = fused.get(memory.id)
              return hit
                ? ({ memory, score: hit.score, signals: hit.signals } as SearchResult)
                : null
            })
            .filter((value): value is SearchResult => value !== null)
            .sort((left, right) => right.score - left.score)
            .slice(0, limit)
        }).pipe(
          Effect.mapError((cause) =>
            cause instanceof StorageError || cause instanceof RetrievalError
              ? cause
              : new RetrievalError({ message: "Failed to search memories", cause })
          )
        )
      return RetrievalService.of({ search })
    })
  )
}

export class AdapterBridgeService extends ServiceMap.Service<
  AdapterBridgeService,
  {
    readonly resolveScope: (scope: Scope | undefined) => Effect.Effect<Scope, InvalidScope>
    readonly recall: (
      query: string,
      scope: Scope | undefined,
      options?: RecallOptions,
      filter?: FilterExpr
    ) => Effect.Effect<ReadonlyArray<SearchResult>, RetrievalError | StorageError | InvalidScope>
    readonly rememberText: (
      text: string,
      scope: Scope | undefined,
      metadata?: JsonRecord
    ) => Effect.Effect<MemoryRecord, StorageError | EmbeddingError | ExtractionError | InvalidScope>
    readonly formatContext: (
      results: ReadonlyArray<SearchResult>,
      options?: RecallOptions
    ) => string
  }
>()("fraction/AdapterBridgeService") {
  static readonly layer = Layer.effect(
    AdapterBridgeService,
    Effect.gen(function* () {
      const memoryService = yield* MemoryService
      const config = yield* FractionConfigRef
      const resolveScope = (scope: Scope | undefined) =>
        Effect.gen(function* () {
          const normalized = normalizeScope(scope, config.defaultNamespace)
          if (!hasScope(normalized)) {
            return yield* new InvalidScope({
              reason: "At least one scope field or default namespace is required"
            })
          }
          return normalized
        })

      const recall = (
        query: string,
        scope: Scope | undefined,
        options?: RecallOptions,
        filter?: FilterExpr
      ) =>
        Effect.gen(function* () {
          const resolved = yield* resolveScope(scope)
          return yield* memoryService.search(
            query,
            resolved,
            filter === undefined
              ? { limit: options?.limit ?? config.topK }
              : { limit: options?.limit ?? config.topK, filter }
          )
        })

      const rememberText = (text: string, scope: Scope | undefined, metadata?: JsonRecord) =>
        Effect.gen(function* () {
          const resolved = yield* resolveScope(scope)
          return yield* memoryService.add(text, resolved, metadata)
        })

      const formatContext = (results: ReadonlyArray<SearchResult>, options?: RecallOptions) =>
        results.length === 0
          ? ""
          : [
              "## Relevant Memory",
              ...results.slice(0, options?.limit ?? config.topK).map((result) => {
                const suffix = options?.includeMetadata
                  ? ` metadata=${JSON.stringify(result.memory.metadata)}`
                  : ""
                return `- ${result.memory.content} [id=${result.memory.id} score=${result.score.toFixed(3)}]${suffix}`
              })
            ].join("\n")

      return AdapterBridgeService.of({ resolveScope, recall, rememberText, formatContext })
    })
  )
}

export class MemoryService extends ServiceMap.Service<
  MemoryService,
  {
    readonly add: (
      input:
        | string
        | ReadonlyArray<{
            readonly role: "user" | "assistant" | "system"
            readonly content: string
          }>,
      scope: Scope,
      metadata?: JsonRecord
    ) => Effect.Effect<MemoryRecord, StorageError | EmbeddingError | ExtractionError>
    readonly addMany: (
      inputs: ReadonlyArray<string>,
      scope: Scope,
      metadata?: JsonRecord
    ) => Effect.Effect<ReadonlyArray<MemoryRecord>, StorageError | EmbeddingError | ExtractionError>
    readonly search: (
      query: string,
      scope: Scope,
      options?: { readonly limit?: number; readonly filter?: FilterExpr | undefined }
    ) => Effect.Effect<ReadonlyArray<SearchResult>, RetrievalError | StorageError>
    readonly get: (id: string) => Effect.Effect<MemoryRecord, StorageError | MemoryNotFound>
    readonly getAll: (scope: Scope) => Effect.Effect<ReadonlyArray<MemoryRecord>, StorageError>
    readonly update: (
      id: string,
      input: string,
      metadata?: JsonRecord
    ) => Effect.Effect<
      MemoryRecord,
      StorageError | MemoryNotFound | EmbeddingError | ExtractionError
    >
    readonly forget: (id: string) => Effect.Effect<void, StorageError | MemoryNotFound>
    readonly delete: (id: string) => Effect.Effect<void, StorageError>
    readonly deleteAll: (scope: Scope) => Effect.Effect<void, StorageError>
    readonly reset: Effect.Effect<void, StorageError>
    readonly history: (id: string) => Effect.Effect<ReadonlyArray<HistoryEvent>, StorageError>
  }
>()("fraction/MemoryService") {
  static readonly layer = Layer.effect(
    MemoryService,
    Effect.gen(function* () {
      const repo = yield* MemoryRepo
      const extractor = yield* ExtractionService
      const embeddings = yield* EmbeddingService
      const config = yield* FractionConfigRef
      const retrieval = yield* RetrievalService

      const recordHistory = (
        rootId: string,
        memoryId: string,
        event: HistoryEvent["event"],
        content: string
      ) =>
        repo.recordHistory({
          id: makeMemoryId(),
          rootId,
          memoryId,
          event,
          content,
          createdAt: nowIso()
        } as HistoryEvent)

      const add = (
        input:
          | string
          | ReadonlyArray<{
              readonly role: "user" | "assistant" | "system"
              readonly content: string
            }>,
        scope: Scope,
        metadata?: JsonRecord
      ) =>
        Effect.gen(function* () {
          const rawContent = messagesToText(input)
          const extracted = yield* extractor.compress(rawContent)
          const contentHash = hashText(extracted.content)
          const existing = yield* repo.listActive(scope)
          const duplicate = existing.find((memory) => hashText(memory.content) === contentHash)
          if (duplicate) {
            yield* recordHistory(duplicate.rootId, duplicate.id, "SKIP", duplicate.content)
            return duplicate
          }
          const createdAt = nowIso()
          const record = createMemoryRecord({
            id: makeMemoryId() as MemoryRecord["id"],
            rootId: makeMemoryId(),
            version: 1,
            content: extracted.content,
            rawContent,
            metadata: metadata ?? {},
            scope,
            createdAt,
            updatedAt: createdAt,
            eventAt: extracted.eventAt,
            isLatest: true
          })
          const embedding = yield* embeddings.embed(record.content)
          yield* repo.insert(record, embedding, extracted.entities)
          yield* recordHistory(record.rootId, record.id, "ADD", record.content)
          return record
        })

      const addMany = (inputs: ReadonlyArray<string>, scope: Scope, metadata?: JsonRecord) =>
        Effect.gen(function* () {
          if (inputs.length === 0) {
            return [] as ReadonlyArray<MemoryRecord>
          }

          const extracted = yield* Effect.forEach(inputs, (input) => extractor.compress(input), {
            concurrency: "unbounded"
          })
          const embeddingsBatch = yield* embeddings.embedMany(
            extracted.map((entry) => entry.content)
          )
          const active = [...(yield* repo.listActive(scope))]
          const records: Array<MemoryRecord> = []

          for (let index = 0; index < inputs.length; index++) {
            const rawContent = inputs[index]!
            const extraction = extracted[index]!
            const embedding = embeddingsBatch[index]!
            const contentHash = hashText(extraction.content)
            const duplicate = active.find((memory) => hashText(memory.content) === contentHash)

            if (duplicate) {
              yield* recordHistory(duplicate.rootId, duplicate.id, "SKIP", duplicate.content)
              records.push(duplicate)
              continue
            }

            const createdAt = nowIso()
            const record = createMemoryRecord({
              id: makeMemoryId() as MemoryRecord["id"],
              rootId: makeMemoryId(),
              version: 1,
              content: extraction.content,
              rawContent,
              metadata: metadata ?? {},
              scope,
              createdAt,
              updatedAt: createdAt,
              eventAt: extraction.eventAt,
              isLatest: true
            })

            yield* repo.insert(record, embedding, extraction.entities)
            yield* recordHistory(record.rootId, record.id, "ADD", record.content)
            active.push(record)
            records.push(record)
          }

          return records
        })

      const search = (
        query: string,
        scope: Scope,
        options?: { readonly limit?: number; readonly filter?: FilterExpr | undefined }
      ) => retrieval.search(query, scope, options?.filter, options?.limit ?? config.topK)

      const get = (id: string) =>
        repo
          .getById(id)
          .pipe(
            Effect.flatMap((record) =>
              record ? Effect.succeed(record) : Effect.fail(new MemoryNotFound({ memoryId: id }))
            )
          )

      const getAll = (scope: Scope) => repo.listAll(scope)

      const update = (id: string, input: string, metadata?: JsonRecord) =>
        Effect.gen(function* () {
          const previous = yield* get(id)
          const extracted = yield* extractor.compress(input)
          const updatedAt = nowIso()
          const next = createMemoryRecord({
            id: makeMemoryId() as MemoryRecord["id"],
            rootId: previous.rootId,
            parentId: previous.id,
            version: previous.version + 1,
            content: extracted.content,
            rawContent: input,
            metadata: metadata ?? previous.metadata,
            scope: previous.scope,
            createdAt: previous.createdAt,
            updatedAt,
            eventAt: extracted.eventAt,
            isLatest: true
          })
          const embedding = yield* embeddings.embed(next.content)
          yield* repo.updateVersion(previous, next, embedding, extracted.entities)
          yield* recordHistory(next.rootId, next.id, "UPDATE", next.content)
          return next
        })

      const forget = (id: string) =>
        Effect.gen(function* () {
          const record = yield* get(id)
          const forgottenAt = nowIso()
          yield* repo.forget(record.id, forgottenAt)
          yield* recordHistory(record.rootId, record.id, "FORGET", record.content)
        })

      const del = (id: string) =>
        Effect.gen(function* () {
          const record = yield* repo.getById(id)
          if (record) {
            yield* recordHistory(record.rootId, record.id, "DELETE", record.content)
          }
          yield* repo.deleteRoot(id)
        })

      const deleteAll = (scope: Scope) =>
        repo.listAll(scope).pipe(
          Effect.flatMap((records) =>
            Effect.gen(function* () {
              for (const record of records) {
                yield* recordHistory(record.rootId, record.id, "DELETE", record.content)
              }
              yield* repo.deleteScope(scope)
            })
          )
        )

      const history = (id: string) => repo.history(id)

      return MemoryService.of({
        add,
        addMany,
        search,
        get,
        getAll,
        update,
        forget,
        delete: del,
        deleteAll,
        reset: repo.reset,
        history
      })
    })
  )
}

export const parseConfig = (input: FractionConfigInput | undefined): FractionConfig => {
  const decoded = (Schema.decodeUnknownSync(FractionConfigSchema)(input ?? {}) ??
    {}) as FractionConfigInput
  return {
    filename: decoded.filename ?? DEFAULT_CONFIG.filename,
    defaultNamespace: decoded.defaultNamespace ?? DEFAULT_CONFIG.defaultNamespace,
    topK: decoded.topK ?? DEFAULT_CONFIG.topK,
    rrfK: decoded.rrfK ?? DEFAULT_CONFIG.rrfK,
    embeddingDimensions: decoded.embeddingDimensions ?? DEFAULT_CONFIG.embeddingDimensions,
    maxFactsPerInput: decoded.maxFactsPerInput ?? DEFAULT_CONFIG.maxFactsPerInput,
    duplicateSimilarity: decoded.duplicateSimilarity ?? DEFAULT_CONFIG.duplicateSimilarity,
    lexicalWeight: decoded.lexicalWeight ?? DEFAULT_CONFIG.lexicalWeight,
    vectorWeight: decoded.vectorWeight ?? DEFAULT_CONFIG.vectorWeight,
    graphWeight: decoded.graphWeight ?? DEFAULT_CONFIG.graphWeight,
    temporalWeight: decoded.temporalWeight ?? DEFAULT_CONFIG.temporalWeight,
    ...(input?.embeddingProvider ? { embeddingProvider: input.embeddingProvider } : {}),
    ...(input?.extractionProvider ? { extractionProvider: input.extractionProvider } : {})
  }
}

export const createFractionLayer = (input?: FractionConfigInput) => {
  const config = parseConfig(input)
  const infraLayer = Layer.mergeAll(
    Layer.succeed(FractionConfigRef, config),
    SqliteClient.layer({ filename: config.filename, disableWAL: false })
  )
  const migrationLayer = MigrationService.layer.pipe(Layer.provide(infraLayer))
  const repoLayer = MemoryRepo.layer.pipe(Layer.provide(infraLayer))
  const embeddingLayer = EmbeddingService.layer.pipe(Layer.provide(infraLayer))
  const extractionLayer = ExtractionService.layer.pipe(Layer.provide(infraLayer))
  const retrievalDeps = Layer.mergeAll(infraLayer, repoLayer, embeddingLayer)
  const retrievalLayer = RetrievalService.layer.pipe(Layer.provide(retrievalDeps))
  const memoryDeps = Layer.mergeAll(
    infraLayer,
    repoLayer,
    embeddingLayer,
    extractionLayer,
    retrievalLayer
  )
  const memoryLayer = MemoryService.layer.pipe(Layer.provide(memoryDeps))
  const bridgeLayer = AdapterBridgeService.layer.pipe(
    Layer.provide(Layer.mergeAll(infraLayer, memoryLayer))
  )

  return Layer.mergeAll(
    infraLayer,
    migrationLayer,
    repoLayer,
    embeddingLayer,
    extractionLayer,
    retrievalLayer,
    memoryLayer,
    bridgeLayer
  )
}

export type FractionRuntime = ManagedRuntime.ManagedRuntime<
  | FractionConfigRef
  | SqlClient.SqlClient
  | MigrationService
  | MemoryRepo
  | EmbeddingService
  | ExtractionService
  | RetrievalService
  | AdapterBridgeService
  | MemoryService,
  MigrationError | StorageError | EmbeddingError | ExtractionError | RetrievalError
>

export const createFractionRuntime = (
  input?: FractionConfigInput,
  options?: { readonly memoMap?: Layer.MemoMap }
) =>
  ManagedRuntime.make(
    createFractionLayer(input) as unknown as Layer.Layer<any, any, never>,
    options
  )
