package bundle

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// LatestPointer names the file that holds the current version, matching
// LATEST_POINTER in ingestion/publish.py.
const LatestPointer = "LATEST"

// Bundle is a fully loaded and verified bundle.
//
// It is immutable once Load returns: nothing in this package writes to it
// afterwards, and callers must not either. That is what makes hot reload
// cheap later -- build a new Bundle in the background, then swap an
// atomic.Pointer[Bundle], with no locks on the request path.
type Bundle struct {
	Version  string
	Dir      string
	Manifest Manifest
	Dim      int
	Vectors  []float32 // Len() x Dim, row-major; see Vector
	Ads      []Ad      // Ads[i] belongs to vector row i
	RowByID  map[string]int
}

// Len is the number of ads (and vectors) in the bundle.
func (b *Bundle) Len() int { return len(b.Ads) }

// Vector returns row i as a slice of the shared backing array. It is not a
// copy, so it must be treated as read-only.
func (b *Bundle) Vector(row int) []float32 {
	return b.Vectors[row*b.Dim : (row+1)*b.Dim]
}

// Load follows root/LATEST to the current bundle and loads it.
func Load(root string) (*Bundle, error) {
	version, err := ResolveLatest(root)
	if err != nil {
		return nil, err
	}
	return LoadDir(filepath.Join(root, version))
}

// ResolveLatest reads the LATEST pointer and returns the version it names.
func ResolveLatest(root string) (string, error) {
	raw, err := os.ReadFile(filepath.Join(root, LatestPointer))
	if err != nil {
		return "", fmt.Errorf("%w: %w", ErrManifest, err)
	}
	version := strings.TrimSpace(string(raw))
	// LATEST is joined onto root, so it must name a directory directly inside
	// it -- never "", "..", or a path somewhere else on disk.
	if !isPlainName(version) {
		return "", fmt.Errorf("%w: LATEST names %q, want a plain version directory", ErrManifest, version)
	}
	return version, nil
}

// LoadDir loads the bundle in one specific version directory.
//
// The order is cheapest check first: the manifest is validated on its own
// before the 68MB vector file is touched, and the vectors are verified before
// the ads are parsed.
func LoadDir(dir string) (*Bundle, error) {
	manifest, err := readManifest(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return nil, err
	}

	vectors, err := readVectors(filepath.Join(dir, manifest.Vectors.File), manifest.Vectors)
	if err != nil {
		return nil, err
	}

	ads, rowByID, err := readAds(filepath.Join(dir, manifest.Ads.File), manifest.Ads)
	if err != nil {
		return nil, err
	}

	return &Bundle{
		Version:  filepath.Base(dir),
		Dir:      dir,
		Manifest: manifest,
		Dim:      manifest.Vectors.Dim,
		Vectors:  vectors,
		Ads:      ads,
		RowByID:  rowByID,
	}, nil
}

// isPlainName reports whether name is a single path element that stays
// inside the directory it is joined onto.
func isPlainName(name string) bool {
	return name != "" && name != "." && name != ".." && !strings.ContainsAny(name, `/\`)
}
