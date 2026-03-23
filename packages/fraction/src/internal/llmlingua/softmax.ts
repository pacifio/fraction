export const softmaxProbability = (
  values: ArrayLike<number>,
  start: number,
  size: number,
  classIndex: number
) => {
  let max = Number.NEGATIVE_INFINITY
  for (let index = 0; index < size; index++) {
    max = Math.max(max, Number(values[start + index] ?? 0))
  }

  let sum = 0
  let selected = 0
  for (let index = 0; index < size; index++) {
    const value = Math.exp(Number(values[start + index] ?? 0) - max)
    if (index === classIndex) {
      selected = value
    }
    sum += value
  }

  return sum === 0 ? 0 : selected / sum
}
