import { Tiktoken } from "js-tiktoken/lite"
import o200kBase from "js-tiktoken/ranks/o200k_base"

import { CompressionUnavailable } from "../errors"

import { loadLlmlinguaSession } from "./loader"
import { softmaxProbability } from "./softmax"
import { getPureToken, isBeginOfNewWord } from "./token-boundaries"
import type {
  LlmlinguaCompressor,
  LlmlinguaPromptOptions,
  LlmlinguaRuntimeOptions,
  LlmlinguaSession
} from "./types"

const budgetTokenizer = new Tiktoken(o200kBase)

const percentile = (values: ReadonlyArray<number>, value: number) => {
  if (values.length === 0) {
    return 0
  }

  const sorted = [...values].sort((left, right) => left - right)
  const clamped = Math.max(0, Math.min(100, value))
  const index = (sorted.length - 1) * (clamped / 100)
  const floor = Math.floor(index)
  const ceil = Math.ceil(index)
  if (floor === ceil) {
    return sorted[floor]!
  }
  const left = sorted[floor]!
  const right = sorted[ceil]!
  return left * (ceil - index) + right * (index - floor)
}

const chunk = <T>(values: ReadonlyArray<T>, size: number) => {
  const batches: Array<Array<T>> = []
  for (let index = 0; index < values.length; index += size) {
    batches.push(values.slice(index, index + size) as Array<T>)
  }
  return batches
}

const toNumberArray = (value: ArrayLike<number | bigint>) =>
  Array.from(value, (item) => (typeof item === "bigint" ? Number(item) : item))

const tokenBudget = (value: string) => budgetTokenizer.encode(value).length

const decodeWords = (session: LlmlinguaSession, words: ReadonlyArray<string>) =>
  words.length === 0 ? "" : session.tokenizer.decoder.decode([...words]).trim()

const placeholderToken = (index: number, family: "bert" | "xlm-roberta") =>
  family === "bert" ? `[unused${index}]` : `<extra_id_${index}>`

const tokenMapForForceTokens = (
  session: LlmlinguaSession,
  forceTokens: ReadonlyArray<string>
) => {
  const tokenMap: Record<string, string> = {}

  for (let index = 0; index < forceTokens.length; index++) {
    const token = forceTokens[index]!
    const tokenized = session.tokenizer.tokenize(token)
    if (tokenized.length === 1) {
      continue
    }

    const placeholder = placeholderToken(index, session.modelFamily)
    if (session.tokenizer.tokenize(placeholder).length === 1) {
      tokenMap[token] = placeholder
    }
  }

  return tokenMap
}

const replaceAddedToken = (value: string, tokenMap: Record<string, string>) => {
  let next = value
  for (const [original, replacement] of Object.entries(tokenMap)) {
    next = next.replaceAll(replacement, original)
  }
  return next
}

const chunkContext = (
  session: LlmlinguaSession,
  originText: string,
  chunkEndTokens: Set<string>
) => {
  const maxLenTokens = session.maxSeqLength - 2
  const originTokens = session.tokenizer.tokenize(originText)
  const chunks: Array<string> = []

  let start = 0
  while (start < originTokens.length) {
    if (start + maxLenTokens > originTokens.length - 1) {
      chunks.push(session.tokenizer.decoder.decode(originTokens.slice(start)).trim())
      break
    }

    let end = start + maxLenTokens
    for (let offset = 0; offset < end - start; offset++) {
      const token = originTokens[end - offset]
      if (token && chunkEndTokens.has(token)) {
        end -= offset
        break
      }
    }

    chunks.push(session.tokenizer.decoder.decode(originTokens.slice(start, end + 1)).trim())
    start = end + 1
  }

  return chunks.filter((entry) => entry.length > 0)
}

const mergeTokenToWord = (
  session: LlmlinguaSession,
  tokens: ReadonlyArray<string>,
  tokenProbs: ReadonlyArray<number>,
  forceTokens: ReadonlyArray<string>,
  tokenMap: Record<string, string>,
  forceReserveDigit: boolean
) => {
  const words: Array<string> = []
  const wordProbsWithForceLogic: Array<Array<number>> = []
  const wordProbsNoForce: Array<Array<number>> = []

  const specialTokens = new Set(session.tokenizer.special_tokens ?? [])

  for (let index = 0; index < tokens.length; index++) {
    const token = tokens[index]!
    let probability = tokenProbs[index] ?? 0
    if (specialTokens.has(token)) {
      continue
    }

    if (isBeginOfNewWord(token, session.modelFamily, forceTokens, tokenMap)) {
      const pureToken = getPureToken(token, session.modelFamily)
      const probabilityNoForce = probability
      if (forceTokens.includes(pureToken) || Object.values(tokenMap).includes(pureToken)) {
        probability = 1
      }
      const restored = replaceAddedToken(token, tokenMap)
      words.push(restored)
      wordProbsWithForceLogic.push([
        forceReserveDigit && /\d/.test(restored) ? 1 : probability
      ])
      wordProbsNoForce.push([probabilityNoForce])
      continue
    }

    const pureToken = getPureToken(token, session.modelFamily)
    if (words.length === 0) {
      words.push(pureToken)
      wordProbsWithForceLogic.push([probability])
      wordProbsNoForce.push([probability])
      continue
    }

    words[words.length - 1] += pureToken
    wordProbsWithForceLogic[wordProbsWithForceLogic.length - 1]!.push(
      forceReserveDigit && /\d/.test(token) ? 1 : probability
    )
    wordProbsNoForce[wordProbsNoForce.length - 1]!.push(probability)
  }

  return { words, wordProbsWithForceLogic, wordProbsNoForce }
}

const tokenProbToWordProb = (
  tokenProbsPerWord: ReadonlyArray<ReadonlyArray<number>>,
  mode: "mean" | "first"
) =>
  tokenProbsPerWord.map((probs) =>
    mode === "first"
      ? (probs[0] ?? 0)
      : probs.reduce((sum, probability) => sum + probability, 0) / Math.max(probs.length, 1)
  )

const normalizeContextLevelRate = (
  contextCount: number,
  originalTokenCount: number,
  rate: number,
  options: {
    readonly targetContext?: number
    readonly contextLevelRate?: number | undefined
    readonly contextLevelTargetToken?: number
    readonly targetToken?: number
  }
) => {
  if (options.targetContext !== undefined && options.targetContext >= 0) {
    return Math.min(options.targetContext / Math.max(contextCount, 1), 1)
  }

  if (options.contextLevelTargetToken !== undefined && options.contextLevelTargetToken >= 0) {
    return Math.min(options.contextLevelTargetToken / Math.max(originalTokenCount, 1), 1)
  }

  if (options.targetToken !== undefined && options.targetToken > 0) {
    return Math.min((options.targetToken * 2) / Math.max(originalTokenCount, 1), 1)
  }

  return options.contextLevelRate ?? Math.min((rate + 1) / 2, 1)
}

const flattenContexts = (contexts: ReadonlyArray<ReadonlyArray<string>>) =>
  contexts.flatMap((entry) => entry)

const extractWordProbabilities = async (
  session: LlmlinguaSession,
  contexts: ReadonlyArray<string>,
  options: {
    readonly tokenToWord: "mean" | "first"
    readonly forceTokens: ReadonlyArray<string>
    readonly tokenMap: Record<string, string>
    readonly forceReserveDigit: boolean
    readonly batchSize: number
  }
) => {
  const batches = chunk(contexts, options.batchSize)
  const results: Array<{ readonly words: ReadonlyArray<string>; readonly probs: ReadonlyArray<number> }> = []

  for (const batch of batches) {
    const encoded = await session.tokenizer(batch, {
      padding: true,
      truncation: true,
      max_length: session.maxSeqLength
    })
    const inputIds = toNumberArray(encoded.input_ids.data)
    const attentionMask = toNumberArray(encoded.attention_mask.data)
    const logits = await session.model({
      input_ids: encoded.input_ids,
      attention_mask: encoded.attention_mask
    })

    const [batchSize = batch.length, seqLength = session.maxSeqLength, classCount = 2] =
      logits.logits.dims
    const logitValues = Array.from(logits.logits.data, Number)

    for (let batchIndex = 0; batchIndex < batchSize; batchIndex++) {
      const rowIds = inputIds.slice(batchIndex * seqLength, (batchIndex + 1) * seqLength)
      const rowMask = attentionMask.slice(batchIndex * seqLength, (batchIndex + 1) * seqLength)
      const tokenProbs: Array<number> = []
      const activeIds: Array<number> = []

      for (let tokenIndex = 0; tokenIndex < seqLength; tokenIndex++) {
        if ((rowMask[tokenIndex] ?? 0) <= 0) {
          continue
        }

        const id = rowIds[tokenIndex]
        if (id === undefined) {
          continue
        }

        activeIds.push(id)
        const start = (batchIndex * seqLength + tokenIndex) * classCount
        tokenProbs.push(softmaxProbability(logitValues, start, classCount, 1))
      }

      const tokenList = session.tokenizer.model.convert_ids_to_tokens(activeIds)
      const { words, wordProbsWithForceLogic, wordProbsNoForce } = mergeTokenToWord(
        session,
        tokenList,
        tokenProbs,
        options.forceTokens,
        options.tokenMap,
        options.forceReserveDigit
      )

      const forcedWordProbabilities = tokenProbToWordProb(
        wordProbsWithForceLogic,
        options.tokenToWord
      )
      const wordProbabilities = tokenProbToWordProb(wordProbsNoForce, options.tokenToWord)

      results.push({
        words,
        probs:
          wordProbabilities.length === forcedWordProbabilities.length
            ? forcedWordProbabilities
            : wordProbabilities
      })
    }
  }

  return results
}

const compressContexts = async (
  session: LlmlinguaSession,
  contexts: ReadonlyArray<string>,
  options: {
    readonly reduceRate: number
    readonly tokenToWord: "mean" | "first"
    readonly forceTokens: ReadonlyArray<string>
    readonly tokenMap: Record<string, string>
    readonly forceReserveDigit: boolean
    readonly dropConsecutive: boolean
    readonly batchSize: number
  }
) => {
  if (options.reduceRate <= 0) {
    return [...contexts]
  }

  const wordProbabilities = await extractWordProbabilities(session, contexts, options)
  const compressed: Array<string> = []

  for (const entry of wordProbabilities) {
    const { words } = entry
    const wordProbs = [...entry.probs]

    if (options.dropConsecutive && words.length > 0) {
      const threshold = percentile(wordProbs, 100 * options.reduceRate)
      let previous: string | undefined
      let tokenBetween = false

      for (let index = 0; index < words.length; index++) {
        const word = words[index]!
        const probability = wordProbs[index]!
        if (options.forceTokens.includes(word)) {
          if (!tokenBetween && previous === word) {
            wordProbs[index] = 0
          }
          tokenBetween = false
          previous = word
        } else {
          tokenBetween ||= probability > threshold
        }
      }
    }

    const expandedProbs: Array<number> = []
    for (let index = 0; index < words.length; index++) {
      const word = words[index]!
      const probability = wordProbs[index] ?? 0
      const length = Math.max(tokenBudget(word), 1)
      expandedProbs.push(...Array.from({ length }, () => probability))
    }

    const threshold = percentile(expandedProbs, 100 * options.reduceRate + 1)
    const keepWords: Array<string> = []

    for (let index = 0; index < words.length; index++) {
      const word = words[index]!
      const probability = wordProbs[index] ?? 0
      if (probability > threshold || (threshold === 1 && probability === threshold)) {
        if (
          options.dropConsecutive &&
          options.forceTokens.includes(word) &&
          keepWords[keepWords.length - 1] === word
        ) {
          continue
        }
        keepWords.push(word)
      }
    }

    compressed.push(replaceAddedToken(decodeWords(session, keepWords), options.tokenMap))
  }

  return compressed
}

const getContextProbabilities = async (
  session: LlmlinguaSession,
  contexts: ReadonlyArray<ReadonlyArray<string>>,
  options: {
    readonly tokenToWord: "mean" | "first"
    readonly forceTokens: ReadonlyArray<string>
    readonly tokenMap: Record<string, string>
    readonly forceReserveDigit: boolean
    readonly batchSize: number
  }
) => {
  const flattened = flattenContexts(contexts)
  const chunkProbabilities = await extractWordProbabilities(session, flattened, options)
  let offset = 0

  return contexts.map((chunks) => {
    const probabilities: Array<number> = []
    for (let index = 0; index < chunks.length; index++) {
      probabilities.push(...(chunkProbabilities[offset + index]?.probs ?? []))
    }
    offset += chunks.length
    return probabilities.length === 0
      ? 0
      : probabilities.reduce((sum, probability) => sum + probability, 0) / probabilities.length
  })
}

const defaultPromptOptions = (options: LlmlinguaPromptOptions | undefined) => ({
  rate: options?.rate ?? 0.6,
  targetToken: options?.targetToken ?? -1,
  tokenToWord: options?.tokenToWord ?? "mean",
  forceTokens: options?.forceTokens ?? [],
  forceReserveDigit: options?.forceReserveDigit ?? false,
  dropConsecutive: options?.dropConsecutive ?? false,
  chunkEndTokens: options?.chunkEndTokens ?? [".", "\n"],
  useContextLevelFilter: options?.useContextLevelFilter ?? false,
  useTokenLevelFilter: options?.useTokenLevelFilter ?? true,
  targetContext: options?.targetContext ?? -1,
  contextLevelRate: options?.contextLevelRate,
  contextLevelTargetToken: options?.contextLevelTargetToken ?? -1,
  forceContextIds: options?.forceContextIds ?? []
})

export const createLlmlingua2Compressor = (
  runtimeOptions: LlmlinguaRuntimeOptions & {
    readonly batchSize?: number
  }
): LlmlinguaCompressor => {
  let sessionPromise: Promise<LlmlinguaSession> | undefined

  const getSession = () => {
    sessionPromise ??= loadLlmlinguaSession(runtimeOptions)
    return sessionPromise
  }

  const compressPrompt = async (
    context: string | ReadonlyArray<string>,
    options?: LlmlinguaPromptOptions
  ) => {
    const settings = defaultPromptOptions(options)
    const session = await getSession()
    const contexts = typeof context === "string" ? [context] : [...context]
    const rawTokenCount = contexts.reduce((sum, entry) => sum + tokenBudget(entry), 0)
    const tokenMap = tokenMapForForceTokens(session, settings.forceTokens)
    const chunkEndTokens = new Set([
      ...settings.chunkEndTokens,
      ...Object.values(tokenMap).filter((token) => settings.chunkEndTokens.includes(token))
    ])

    const replacedContexts = contexts.map((entry) => {
      let next = entry
      for (const [original, replacement] of Object.entries(tokenMap)) {
        next = next.replaceAll(original, replacement)
      }
      return next
    })

    const chunkedContexts = replacedContexts.map((entry) => chunkContext(session, entry, chunkEndTokens))

    let reduceRate =
      settings.targetToken > 0 && rawTokenCount > 0
        ? 1 - Math.min(settings.targetToken / rawTokenCount, 1)
        : 1 - settings.rate

    let selectedContexts = chunkedContexts

    if (settings.useContextLevelFilter && chunkedContexts.length > 1) {
      const contextLevelRate = normalizeContextLevelRate(
        chunkedContexts.length,
        rawTokenCount,
        settings.rate,
        settings
      )
      const contextProbabilities = await getContextProbabilities(session, chunkedContexts, {
        tokenToWord: settings.tokenToWord,
        forceTokens: settings.forceTokens,
        tokenMap,
        forceReserveDigit: settings.forceReserveDigit,
        batchSize: runtimeOptions.batchSize ?? 16
      })
      const threshold = percentile(contextProbabilities, 100 * (1 - contextLevelRate))
      selectedContexts = chunkedContexts.filter(
        (_chunks, index) =>
          contextProbabilities[index]! >= threshold || settings.forceContextIds.includes(index)
      )

      const selectedTokenBudget = flattenContexts(selectedContexts).reduce(
        (sum, entry) => sum + tokenBudget(entry),
        0
      )
      if (settings.targetToken > 0 && selectedTokenBudget > 0) {
        reduceRate = 1 - Math.min(settings.targetToken / selectedTokenBudget, 1)
      }
    }

    const compressedContexts = settings.useTokenLevelFilter
      ? await compressContexts(session, flattenContexts(selectedContexts), {
          reduceRate: Math.max(0, reduceRate),
          tokenToWord: settings.tokenToWord,
          forceTokens: settings.forceTokens,
          tokenMap,
          forceReserveDigit: settings.forceReserveDigit,
          dropConsecutive: settings.dropConsecutive,
          batchSize: runtimeOptions.batchSize ?? 16
        })
      : flattenContexts(selectedContexts)

    const content = compressedContexts.join("\n\n").trim()
    const tokenCountAfter = tokenBudget(content)

    return {
      content,
      tokenCountBefore: rawTokenCount,
      tokenCountAfter,
      retainedRatio: rawTokenCount === 0 ? 1 : tokenCountAfter / rawTokenCount
    }
  }

  return {
    compress: async (text, options) => compressPrompt(text, options),
    compressPrompt,
    close: async () => {
      if (!sessionPromise) {
        return
      }
      try {
        const session = await sessionPromise
        await session.dispose()
      } catch (cause) {
        throw new CompressionUnavailable({
          message: "Failed to dispose LLMLingua-2 model resources",
          model: runtimeOptions.model,
          cause
        })
      } finally {
        sessionPromise = undefined
      }
    }
  }
}

export const prefetchLlmlinguaArtifacts = async (options: LlmlinguaRuntimeOptions) => {
  const session = await loadLlmlinguaSession(options)
  await session.dispose()
}
