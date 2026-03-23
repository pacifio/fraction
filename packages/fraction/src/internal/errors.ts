import { Schema } from "effect"

export class MemoryNotFound extends Schema.TaggedErrorClass<MemoryNotFound>()("MemoryNotFound", {
  memoryId: Schema.String
}) {}

export class InvalidScope extends Schema.TaggedErrorClass<InvalidScope>()("InvalidScope", {
  reason: Schema.String
}) {}

export class StorageError extends Schema.TaggedErrorClass<StorageError>()("StorageError", {
  message: Schema.String,
  cause: Schema.optional(Schema.Defect)
}) {}

export class MigrationError extends Schema.TaggedErrorClass<MigrationError>()("MigrationError", {
  message: Schema.String,
  cause: Schema.optional(Schema.Defect)
}) {}

export class ExtractionError extends Schema.TaggedErrorClass<ExtractionError>()("ExtractionError", {
  message: Schema.String,
  cause: Schema.optional(Schema.Defect)
}) {}

export class CompressionError extends Schema.TaggedErrorClass<CompressionError>()(
  "CompressionError",
  {
    message: Schema.String,
    cause: Schema.optional(Schema.Defect)
  }
) {}

export class CompressionUnavailable extends Schema.TaggedErrorClass<CompressionUnavailable>()(
  "CompressionUnavailable",
  {
    message: Schema.String,
    model: Schema.optional(Schema.String),
    cause: Schema.optional(Schema.Defect)
  }
) {}

export class EmbeddingError extends Schema.TaggedErrorClass<EmbeddingError>()("EmbeddingError", {
  message: Schema.String,
  cause: Schema.optional(Schema.Defect)
}) {}

export class RetrievalError extends Schema.TaggedErrorClass<RetrievalError>()("RetrievalError", {
  message: Schema.String,
  cause: Schema.optional(Schema.Defect)
}) {}

export class AdapterError extends Schema.TaggedErrorClass<AdapterError>()("AdapterError", {
  message: Schema.String,
  cause: Schema.optional(Schema.Defect)
}) {}
