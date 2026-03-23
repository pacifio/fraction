const PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

export const inferLlmlinguaModelFamily = (
  modelType: string | null | undefined,
  modelName: string
): "bert" | "xlm-roberta" => {
  const normalized = `${modelType ?? ""} ${modelName}`.toLowerCase()
  if (normalized.includes("xlm-roberta")) {
    return "xlm-roberta"
  }
  return "bert"
}

export const getPureToken = (token: string | null | undefined, family: "bert" | "xlm-roberta") => {
  if (!token) {
    return ""
  }
  return family === "xlm-roberta" ? token.replace(/^▁/, "") : token.replace(/^##/, "")
}

export const isBeginOfNewWord = (
  token: string | null | undefined,
  family: "bert" | "xlm-roberta",
  forceTokens: ReadonlyArray<string>,
  tokenMap: Record<string, string>
) => {
  if (!token) {
    return false
  }
  if (family === "xlm-roberta") {
    if (
      PUNCTUATION.includes(token) ||
      forceTokens.includes(token) ||
      Object.values(tokenMap).includes(token)
    ) {
      return true
    }
    return token.startsWith("▁")
  }

  const normalized = token.replace(/^##/, "")
  if (forceTokens.includes(normalized) || Object.values(tokenMap).includes(normalized)) {
    return true
  }
  return !token.startsWith("##")
}
