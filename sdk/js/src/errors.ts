/**
 * Base error class for all mind-mem SDK errors.
 */
export class MindMemError extends Error {
  readonly statusCode: number;
  readonly responseBody: unknown;

  constructor(message: string, statusCode: number, responseBody: unknown = null) {
    super(message);
    this.name = "MindMemError";
    this.statusCode = statusCode;
    this.responseBody = responseBody;
    // Restore prototype chain (required when extending built-ins in TS)
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * Thrown when the server returns 401 or 403.
 * Check that the token is set and valid.
 */
export class MindMemAuthError extends MindMemError {
  constructor(message: string, statusCode: 401 | 403, responseBody: unknown = null) {
    super(message, statusCode, responseBody);
    this.name = "MindMemAuthError";
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * Thrown when the server returns 429 Too Many Requests.
 * Inspect `retryAfterSeconds` before retrying.
 */
export class MindMemRateLimitError extends MindMemError {
  readonly retryAfterSeconds: number | null;

  constructor(
    message: string,
    retryAfterSeconds: number | null,
    responseBody: unknown = null,
  ) {
    super(message, 429, responseBody);
    this.name = "MindMemRateLimitError";
    this.retryAfterSeconds = retryAfterSeconds;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * Thrown when the server returns a 5xx status code.
 */
export class MindMemServerError extends MindMemError {
  constructor(message: string, statusCode: number, responseBody: unknown = null) {
    super(message, statusCode, responseBody);
    this.name = "MindMemServerError";
    Object.setPrototypeOf(this, new.target.prototype);
  }
}
