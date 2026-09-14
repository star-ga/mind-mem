// Copyright 2026 STARGA, Inc.
package mindmem_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	mindmem "github.com/star-ga/mind-mem/sdk/go/v5"
)

func contractServer(t *testing.T, name string) (*mindmem.Client, func()) {
	t.Helper()
	raw, err := os.ReadFile("testdata/contract/" + name + ".json")
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(raw)
	}))
	return mindmem.NewClient(srv.URL), srv.Close
}

func TestActualServerRecallContract(t *testing.T) {
	c, close := contractServer(t, "recall")
	defer close()
	result, err := c.Recall(context.Background(), "orchid", mindmem.RecallOptions{ScoringInstant: "2026-09-14"})
	if err != nil {
		t.Fatal(err)
	}
	if result.Count != 1 || len(result.Results) != 1 || result.Results[0].ID != "D-20260914-001" || result.Results[0].Excerpt != "Orchid is the SDK contract sentinel." {
		t.Fatalf("lost served result: %+v", result)
	}
	if result.SchemaVersion != "1.0" || result.ScoringInstant != "2026-09-14" {
		t.Fatalf("lost response identity: %+v", result)
	}
	var attestation map[string]json.RawMessage
	if err := json.Unmarshal(result.Attestation, &attestation); err != nil {
		t.Fatal(err)
	}
	if string(attestation["schema"]) != `"RECALL_ATTEST_v2"` {
		t.Fatal("lost attestation")
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(result.Raw, &raw); err != nil {
		t.Fatal(err)
	}
	if string(raw["attestation"]) != string(result.Attestation) {
		t.Fatal("raw evidence changed")
	}
}

func TestActualServerBlockContract(t *testing.T) {
	c, close := contractServer(t, "block")
	defer close()
	result, err := c.GetBlock(context.Background(), "D-20260914-001")
	if err != nil {
		t.Fatal(err)
	}
	if !result.Found || result.Block.ID != result.BlockID || result.Block.Statement != "Orchid is the SDK contract sentinel." || result.Block.Status != "active" {
		t.Fatalf("lost governed fields: %+v", result)
	}
	if len(result.Raw) == 0 {
		t.Fatal("raw optional fields were discarded")
	}
}

func TestActualServerDiagnosticContracts(t *testing.T) {
	c, close := contractServer(t, "contradictions")
	contradictions, err := c.ListContradictions(context.Background())
	close()
	if err != nil {
		t.Fatal(err)
	}
	if contradictions.Status != "clean" || contradictions.Contradictions != 0 {
		t.Fatalf("wrong count envelope: %+v", contradictions)
	}
	c, close = contractServer(t, "health")
	health, err := c.Health(context.Background())
	close()
	if err != nil {
		t.Fatal(err)
	}
	if health.WorkspaceExists == nil || !*health.WorkspaceExists || health.Workspace == nil || health.APIVersion == "" || health.WorkspaceSchemaVersion == "" {
		t.Fatalf("lost health fields: %+v", health)
	}
	c, close = contractServer(t, "scan")
	scan, err := c.Scan(context.Background())
	close()
	if err != nil {
		t.Fatal(err)
	}
	var counts map[string]int
	if err := json.Unmarshal(scan.Checks["decisions"], &counts); err != nil {
		t.Fatal(err)
	}
	if counts["active"] != 1 || scan.Backend != "markdown" {
		t.Fatalf("lost scan checks: %+v", scan)
	}
}

func TestPublicHealthPreservesAbsence(t *testing.T) {
	c, close := contractServer(t, "health_public")
	defer close()
	health, err := c.Health(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if health.APIVersion == "" || health.WorkspaceExists != nil || health.Workspace != nil {
		t.Fatalf("public health must omit private workspace fields: %+v", health)
	}
}
