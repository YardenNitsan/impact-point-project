import { createHash, randomBytes, timingSafeEqual } from "node:crypto";

export const SIMULATION_TOKEN_HEADER = "x-simulation-token";
const TOKEN_BYTES = 32;

export function hashSimulationAccessToken(token: string): string {
  return createHash("sha256").update(token.trim()).digest("hex");
}

export function generateSimulationAccessToken(): {
  token: string;
  tokenHash: string;
} {
  const token = randomBytes(TOKEN_BYTES).toString("hex");

  return {
    token,
    tokenHash: hashSimulationAccessToken(token),
  };
}

export function simulationAccessTokenMatches(
  providedToken: string,
  storedTokenHash: string,
): boolean {
  const providedHash = hashSimulationAccessToken(providedToken);

  const providedBuffer = Buffer.from(providedHash, "hex");
  const storedBuffer = Buffer.from(storedTokenHash, "hex");

  if (providedBuffer.length !== storedBuffer.length) {
    return false;
  }

  return timingSafeEqual(providedBuffer, storedBuffer);
}
