package mindmem

import (
	"net/url"
	"strings"
)

// Route is one REST operation this client knows how to call, expressed in the
// same (method, path-template) form the OpenAPI document uses.
//
// The table exists so the routes are DATA rather than string literals buried
// in call sites. `sdk/spec/openapi.json` is the contract; the conformance gate
// in `tests/test_sdk_route_conformance.py` reads the literals below and fails
// when any of them names an operation the server does not serve. That gate is
// the reason Method and Path are plain string literals and not http.MethodPost
// constants: a cross-language checker has to be able to read them without a Go
// toolchain.
//
// Before this table existed the client called `GET /v1/recall` with query
// parameters and `GET /v1/blocks/{id}`, while the server has served
// `POST /v1/recall` with a JSON body and `GET /v1/block/{block_id}` — two
// endpoints that could never have answered.
type Route struct {
	// Method is the uppercase HTTP verb.
	Method string
	// Path is the OpenAPI path template, placeholders included, e.g.
	// "/v1/block/{block_id}".
	Path string
}

// Route table. Every method in methods.go resolves its request through one of
// these; TestRoutes_MethodsUseTheDeclaredRoutes proves it.
var (
	RouteRecall             = Route{Method: "POST", Path: "/v1/recall"}
	RouteGetBlock           = Route{Method: "GET", Path: "/v1/block/{block_id}"}
	RouteListContradictions = Route{Method: "GET", Path: "/v1/contradictions"}
	RouteHealth             = Route{Method: "GET", Path: "/v1/health"}
	RouteScan               = Route{Method: "GET", Path: "/v1/scan"}
)

// Routes is every operation this client can issue, in declaration order.
func Routes() []Route {
	return []Route{
		RouteRecall,
		RouteGetBlock,
		RouteListContradictions,
		RouteHealth,
		RouteScan,
	}
}

// Expand substitutes the route's path placeholders, left to right, with the
// path-escaped args. It panics on an arity mismatch because that is a
// programming error in this package, never a caller's runtime input: the
// number of placeholders in a route is fixed at compile time.
func (r Route) Expand(args ...string) string {
	path := r.Path
	for _, arg := range args {
		open := strings.IndexByte(path, '{')
		if open < 0 {
			panic("mind-mem: too many arguments for route " + r.Path)
		}
		closeIdx := strings.IndexByte(path[open:], '}')
		if closeIdx < 0 {
			panic("mind-mem: malformed route template " + r.Path)
		}
		path = path[:open] + url.PathEscape(arg) + path[open+closeIdx+1:]
	}
	if strings.ContainsAny(path, "{}") {
		panic("mind-mem: too few arguments for route " + r.Path)
	}
	return path
}
