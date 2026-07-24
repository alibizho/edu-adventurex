const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly detail?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function apiBaseUrl() {
  return (import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL).replace(/\/$/, "");
}

export async function apiRequest<T>(
  path: string,
  init: RequestInit = {},
  timeoutMs = 30_000,
): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${apiBaseUrl()}${path}`, {
      ...init,
      signal: controller.signal,
      headers: init.body instanceof FormData
        ? init.headers
        : { "Content-Type": "application/json", ...init.headers },
    });
    const contentType = response.headers.get("content-type") ?? "";
    const payload = contentType.includes("application/json")
      ? await response.json()
      : await response.text();
    if (!response.ok) {
      const detail = typeof payload === "object" && payload && "detail" in payload
        ? (payload as { detail: unknown }).detail
        : payload;
      throw new ApiError(
        typeof detail === "string" ? detail : `REQUEST FAILED (${response.status})`,
        response.status,
        detail,
      );
    }
    return payload as T;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiError("THE BACKEND REQUEST TIMED OUT.", 408);
    }
    throw new ApiError(
      `CANNOT REACH THE BACKEND AT ${apiBaseUrl()}. CHECK THAT IT IS RUNNING.`,
      0,
      error,
    );
  } finally {
    window.clearTimeout(timeout);
  }
}

export function apiMessage(error: unknown) {
  return error instanceof Error ? error.message.toUpperCase() : "UNEXPECTED BACKEND ERROR.";
}
