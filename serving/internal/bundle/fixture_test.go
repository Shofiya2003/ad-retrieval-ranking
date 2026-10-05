package bundle

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fixture is a tiny in-memory bundle that a test can damage in one specific
// way before (or after) it is written to disk.
type fixture struct {
	count   int
	dim     int
	vectors []float32
	adLines []string

	// editManifest runs after the real checksums are computed, so a test can
	// make the manifest lie about otherwise-valid files.
	editManifest func(*Manifest)
	// after runs once everything is on disk, for damage to the files themselves.
	after func(t *testing.T, root, dir string)
}

const fixtureVersion = "20260101T000000Z-test0001"

// newFixture builds count deterministic unit vectors of length dim, and one
// valid ad per row.
func newFixture(count, dim int) *fixture {
	f := &fixture{count: count, dim: dim, vectors: make([]float32, count*dim)}
	for row := 0; row < count; row++ {
		v := f.vectors[row*dim : (row+1)*dim]
		var sumSq float64
		for j := range v {
			x := math.Sin(float64(row*dim + j + 1))
			v[j] = float32(x)
			sumSq += x * x
		}
		norm := math.Sqrt(sumSq)
		for j := range v {
			v[j] = float32(float64(v[j]) / norm)
		}
		f.adLines = append(f.adLines, adLine(row, fmt.Sprintf("ad_%04d", row)))
	}
	return f
}

func adLine(row int, adID string) string {
	line, _ := json.Marshal(Ad{
		Row:          row,
		AdID:         adID,
		Headline:     fmt.Sprintf("Test headline number %d", row),
		Category:     "electronics",
		AdvertiserID: "adv_0001",
		BidCPMUSD:    2.5,
		CreatedAt:    "2026-01-01T00:00:00Z",
	})
	return string(line)
}

func (f *fixture) setRow(row int, values ...float32) {
	v := f.vectors[row*f.dim : (row+1)*f.dim]
	for j := range v {
		v[j] = values[j%len(values)]
	}
}

// write lays the bundle out exactly as ingestion/publish.py does and returns
// the root directory (the one holding LATEST).
func (f *fixture) write(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	dir := filepath.Join(root, fixtureVersion)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}

	vecBytes := make([]byte, 4*len(f.vectors))
	for i, x := range f.vectors {
		binary.LittleEndian.PutUint32(vecBytes[i*4:], math.Float32bits(x))
	}
	adBytes := []byte(strings.Join(f.adLines, "\n") + "\n")

	m := Manifest{
		FormatVersion: FormatVersion,
		CreatedAt:     "2026-01-01T00:00:00Z",
		Model: ModelSpec{
			Name: "test-model", Revision: "abc123", Dim: f.dim,
			MaxTokens: 256, Normalized: true, Similarity: "dot",
		},
		Vectors: VectorsSpec{
			File: "vectors.f32", DType: "float32", ByteOrder: "little", Layout: "row_major",
			Count: f.count, Dim: f.dim, Bytes: int64(len(vecBytes)), SHA256: sha(vecBytes),
		},
		Ads: AdsSpec{File: "ads.jsonl", Count: f.count, SHA256: sha(adBytes)},
	}
	if f.editManifest != nil {
		f.editManifest(&m)
	}
	manifestBytes, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		t.Fatal(err)
	}

	writeFile(t, filepath.Join(dir, "vectors.f32"), vecBytes)
	writeFile(t, filepath.Join(dir, "ads.jsonl"), adBytes)
	writeFile(t, filepath.Join(dir, "manifest.json"), manifestBytes)
	writeFile(t, filepath.Join(root, LatestPointer), []byte(fixtureVersion+"\n"))

	if f.after != nil {
		f.after(t, root, dir)
	}
	return root
}

func sha(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func writeFile(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}
