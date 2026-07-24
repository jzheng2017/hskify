export async function sha256Hex(
  bytes: ArrayBuffer,
  cryptoApi: Crypto = globalThis.crypto,
): Promise<string> {
  const digest = await cryptoApi.subtle.digest('SHA-256', bytes)
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('')
}
