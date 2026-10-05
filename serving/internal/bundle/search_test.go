package bundle

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"testing"
)

// parityTolerance allows for float32 rounding. numpy and this Go loop add the
// 384 products in a different order, so the last digits of a score can differ
// without either being wrong.
const parityTolerance = 1e-5

func TestTopKMatchesFullSort(t *testing.T) {
	b, err := Load(newFixture(50, 8).write(t))
	if err != nil {
		t.Fatal(err)
	}
	query := b.Vector(17)

	// The obvious, slow answer: score everything and sort it all.
	all := make([]Hit, b.Len())
	for row := range all {
		all[row] = Hit{Row: row, Score: dot(query, b.Vector(row))}
	}
	sort.Slice(all, func(i, j int) bool { return worse(all[j], all[i]) })

	for _, k := range []int{1, 5, 50} {
		got, err := b.TopK(query, k)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != k {
			t.Fatalf("k=%d: got %d hits", k, len(got))
		}
		for i := range got {
			if got[i] != all[i] {
				t.Fatalf("k=%d position %d: got %+v, want %+v", k, i, got[i], all[i])
			}
		}
	}
}

func TestTopKEdgeCases(t *testing.T) {
	b, err := Load(newFixture(10, 8).write(t))
	if err != nil {
		t.Fatal(err)
	}

	if hits, _ := b.TopK(b.Vector(0), 100); len(hits) != 10 {
		t.Errorf("k larger than the bundle: got %d hits, want all 10", len(hits))
	}
	if hits, _ := b.TopK(b.Vector(0), 0); len(hits) != 0 {
		t.Errorf("k=0: got %d hits, want none", len(hits))
	}
	if _, err := b.TopK(make([]float32, 7), 3); err == nil {
		t.Error("query of the wrong dimension: want an error, got none")
	}
}

// --- tests against the real published bundle --------------------------------
//
// These need `make pipeline` to have run. They skip on a fresh clone (the data
// is gitignored) so `go test` still passes there.

var (
	realOnce   sync.Once
	realBundle *Bundle
	realErr    error
)

func publishedRoot() string {
	if root := os.Getenv("BUNDLE_ROOT"); root != "" {
		return root
	}
	return filepath.Join("..", "..", "..", "data", "published")
}

func groundTruthPath() string {
	if path := os.Getenv("GROUND_TRUTH"); path != "" {
		return path
	}
	return filepath.Join("..", "..", "..", "data", "eval", "ground_truth.json")
}

// loadReal loads the real bundle once and shares it between tests: it is
// 44,200 x 384 floats, and there is no reason to verify it more than once.
func loadReal(tb testing.TB) *Bundle {
	tb.Helper()
	root := publishedRoot()
	if _, err := os.Stat(filepath.Join(root, LatestPointer)); errors.Is(err, os.ErrNotExist) {
		tb.Skipf("no published bundle under %s -- run `make pipeline` first", root)
	}
	realOnce.Do(func() { realBundle, realErr = Load(root) })
	if realErr != nil {
		tb.Fatalf("Load(%s): %v", root, realErr)
	}
	return realBundle
}

// An ad's own vector, used as the query, must find that same ad first with a
// score of ~1.0 (a unit vector dotted with itself). A wrong byte order or an
// off-by-one row offset would break this immediately -- and it needs nothing
// from Python.
func TestRealBundleSelfMatch(t *testing.T) {
	b := loadReal(t)
	n := b.Len()
	for _, row := range []int{0, 1, n / 4, n / 2, 3 * n / 4, n - 2, n - 1} {
		hits, err := b.TopK(b.Vector(row), 1)
		if err != nil {
			t.Fatal(err)
		}
		if hits[0].Row != row {
			t.Errorf("row %d (%s): top hit is row %d", row, b.Ads[row].AdID, hits[0].Row)
		}
		if math.Abs(float64(hits[0].Score)-1) > NormTolerance {
			t.Errorf("row %d: self-similarity %.6f, want ~1.0", row, hits[0].Score)
		}
	}
}

type groundTruth struct {
	BundleVersion string `json:"bundle_version"`
	VectorsSHA256 string `json:"vectors_sha256"`
	K             int    `json:"k"`
	Queries       []struct {
		Query  string    `json:"query"`
		Vector []float32 `json:"vector"`
		TopK   []struct {
			AdID  string  `json:"ad_id"`
			Row   int     `json:"row"`
			Score float64 `json:"score"`
		} `json:"top_k"`
	} `json:"queries"`
}

// Go's exact search must return what Python's exact search returned for the
// same query vectors. Agreement proves the two languages read the same bytes
// as the same numbers, paired with the same ads.
func TestRealBundleMatchesPython(t *testing.T) {
	b := loadReal(t)

	raw, err := os.ReadFile(groundTruthPath())
	if errors.Is(err, os.ErrNotExist) {
		t.Skipf("no ground truth at %s -- run `make ground-truth` first", groundTruthPath())
	}
	if err != nil {
		t.Fatal(err)
	}
	var gt groundTruth
	if err := json.Unmarshal(raw, &gt); err != nil {
		t.Fatal(err)
	}

	// Stale ground truth is a FAILURE, not a skip: comparing old answers
	// against new vectors would report nonsense either way.
	if gt.BundleVersion != b.Version || gt.VectorsSHA256 != b.Manifest.Vectors.SHA256 {
		t.Fatalf("ground truth is for bundle %s (vectors %.12s) but LATEST is %s (vectors %.12s) "+
			"-- run `make ground-truth`", gt.BundleVersion, gt.VectorsSHA256, b.Version, b.Manifest.Vectors.SHA256)
	}
	if len(gt.Queries) == 0 {
		t.Fatal("ground truth has no queries")
	}

	for _, q := range gt.Queries {
		t.Run(q.Query, func(t *testing.T) {
			hits, err := b.TopK(q.Vector, len(q.TopK))
			if err != nil {
				t.Fatal(err)
			}
			if len(hits) != len(q.TopK) {
				t.Fatalf("Go returned %d hits, Python %d", len(hits), len(q.TopK))
			}

			// Position by position, the scores must agree. Both lists are
			// sorted best first, so this also checks the ordering.
			for i, want := range q.TopK {
				if diff := math.Abs(float64(hits[i].Score) - want.Score); diff > parityTolerance {
					t.Errorf("position %d: Go score %.7f, Python %.7f", i, hits[i].Score, want.Score)
				}
				// The row -> ad_id pairing must agree across languages too.
				if b.Ads[want.Row].AdID != want.AdID {
					t.Errorf("Python row %d is %s, Go row %d is %s", want.Row, want.AdID, want.Row, b.Ads[want.Row].AdID)
				}
			}

			// The rows may differ only by near-ties: an ad Python included
			// that Go left out must score within tolerance of Go's cut-off.
			gotRows := make(map[int]bool, len(hits))
			for _, h := range hits {
				gotRows[h.Row] = true
			}
			cutoff := float64(hits[len(hits)-1].Score)
			for _, want := range q.TopK {
				if gotRows[want.Row] {
					continue
				}
				score := float64(dot(q.Vector, b.Vector(want.Row)))
				if cutoff-score > parityTolerance {
					t.Errorf("Python returned row %d (%s, score %.7f) but Go ranked it below its cut-off %.7f",
						want.Row, want.AdID, score, cutoff)
				}
			}
		})
	}
}

func BenchmarkLoad(b *testing.B) {
	root := publishedRoot()
	if _, err := os.Stat(filepath.Join(root, LatestPointer)); err != nil {
		b.Skipf("no published bundle under %s", root)
	}
	for i := 0; i < b.N; i++ {
		if _, err := Load(root); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTopK(b *testing.B) {
	bundle := loadReal(b)
	query := bundle.Vector(0)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := bundle.TopK(query, 10); err != nil {
			b.Fatal(err)
		}
	}
}
