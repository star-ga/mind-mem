/**
 * REST route table.
 *
 * The routes are DATA rather than string literals buried in call sites so a
 * single artifact — `sdk/spec/openapi.json` — can be the contract for both
 * in-tree clients. `tests/test_sdk_route_conformance.py` reads the literals
 * below and fails when any of them names an operation the server does not
 * serve; that gate is why `method` and `path` are plain literals a
 * cross-language checker can read without a TypeScript toolchain.
 *
 * Before this table existed the client issued `GET /v1/recall` with query
 * parameters and `GET /v1/blocks/{id}`, while the server has served
 * `POST /v1/recall` with a JSON body and `GET /v1/block/{block_id}` — two
 * endpoints that could never have answered.
 */

export type HttpMethod = "GET" | "POST" | "DELETE";

export interface Route {
  /** Uppercase HTTP verb. */
  readonly method: HttpMethod;
  /** OpenAPI path template, placeholders included. */
  readonly path: string;
}

export const ROUTE_RECALL: Route = { method: "POST", path: "/v1/recall" };
export const ROUTE_GET_BLOCK: Route = { method: "GET", path: "/v1/block/{block_id}" };
export const ROUTE_LIST_CONTRADICTIONS: Route = { method: "GET", path: "/v1/contradictions" };
export const ROUTE_HEALTH: Route = { method: "GET", path: "/v1/health" };
export const ROUTE_SCAN: Route = { method: "GET", path: "/v1/scan" };

/** Every operation this client can issue, in declaration order. */
export const ROUTES: readonly Route[] = [
  ROUTE_RECALL,
  ROUTE_GET_BLOCK,
  ROUTE_LIST_CONTRADICTIONS,
  ROUTE_HEALTH,
  ROUTE_SCAN,
];

/**
 * Substitute a route's path placeholders, left to right, with URI-encoded
 * args. Throws on an arity mismatch rather than shipping a literal
 * `{block_id}` to the server.
 */
export function expandRoute(route: Route, ...args: readonly string[]): string {
  let path = route.path;
  for (const arg of args) {
    const open = path.indexOf("{");
    if (open < 0) {
      throw new RangeError(`mind-mem: too many arguments for route ${route.path}`);
    }
    const close = path.indexOf("}", open);
    if (close < 0) {
      throw new RangeError(`mind-mem: malformed route template ${route.path}`);
    }
    path = path.slice(0, open) + encodeURIComponent(arg) + path.slice(close + 1);
  }
  if (path.includes("{") || path.includes("}")) {
    throw new RangeError(`mind-mem: too few arguments for route ${route.path}`);
  }
  return path;
}
