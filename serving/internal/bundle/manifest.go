// Package bundle loads the versioned bundle that ingestion.publish writes.
//
// The bundle is untrusted input. Every claim the manifest makes is checked
// against the bytes on disk, and any mismatch is a hard error: a service that
// refuses to start is far easier to notice than one that serves vectors paired
// with the wrong ads.
package bundle

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
)

// FormatVersion is the only bundle layout this loader understands. An
// unknown version fails closed rather than being guessed at.
const FormatVersion = 1

// Sentinel errors, wrapped with %w, so callers can tell failure kinds apart
// with errors.Is without parsing messages.
var (
	ErrManifest = errors.New("bad manifest")
	ErrChecksum = errors.New("checksum mismatch")
	ErrVectors  = errors.New("bad vectors")
	ErrAds      = errors.New("bad ads")
)

// Manifest mirrors manifest.json as written by ingestion/publish.py.
type Manifest struct {
	FormatVersion int         `json:"format_version"`
	CreatedAt     string      `json:"created_at"`
	Model         ModelSpec   `json:"model"`
	Vectors       VectorsSpec `json:"vectors"`
	Ads           AdsSpec     `json:"ads"`
	Source        SourceSpec  `json:"source"`
}

// ModelSpec is the identity of the model that produced the vectors. The query
// sidecar must embed with exactly this model, or similarity scores are
// meaningless.
type ModelSpec struct {
	Name       string `json:"name"`
	Revision   string `json:"revision"`
	Dim        int    `json:"dim"`
	MaxTokens  int    `json:"max_tokens"`
	Normalized bool   `json:"normalized"`
	Similarity string `json:"similarity"`
}

type VectorsSpec struct {
	File      string `json:"file"`
	DType     string `json:"dtype"`
	ByteOrder string `json:"byte_order"`
	Layout    string `json:"layout"`
	Count     int    `json:"count"`
	Dim       int    `json:"dim"`
	Bytes     int64  `json:"bytes"`
	SHA256    string `json:"sha256"`
}

type AdsSpec struct {
	File   string `json:"file"`
	Count  int    `json:"count"`
	SHA256 string `json:"sha256"`
}

type SourceSpec struct {
	CleanFile   string `json:"clean_file"`
	CleanSHA256 string `json:"clean_sha256"`
	EmbeddedAt  string `json:"embedded_at"`
	Device      string `json:"device"`
}

func readManifest(path string) (Manifest, error) {
	var m Manifest
	raw, err := os.ReadFile(path)
	if err != nil {
		return m, fmt.Errorf("%w: %w", ErrManifest, err)
	}
	if err := json.Unmarshal(raw, &m); err != nil {
		return m, fmt.Errorf("%w: %s: %v", ErrManifest, path, err)
	}
	return m, m.validate()
}

// validate checks everything the manifest can be checked for on its own,
// BEFORE any large file is read. Cheap checks first: there is no point
// hashing 68MB of vectors described by a manifest that contradicts itself.
func (m *Manifest) validate() error {
	fail := func(format string, args ...any) error {
		return fmt.Errorf("%w: "+format, append([]any{ErrManifest}, args...)...)
	}

	if m.FormatVersion != FormatVersion {
		return fail("format_version %d, this loader only reads %d", m.FormatVersion, FormatVersion)
	}

	v := m.Vectors
	if v.DType != "float32" || v.ByteOrder != "little" || v.Layout != "row_major" {
		return fail("vectors are %s/%s/%s, want float32/little/row_major", v.DType, v.ByteOrder, v.Layout)
	}

	// Search scores with a plain dot product. That is only cosine similarity
	// if every vector is unit length, so the manifest must promise both.
	if !m.Model.Normalized || m.Model.Similarity != "dot" {
		return fail("model similarity %q normalized=%v, want dot over unit vectors",
			m.Model.Similarity, m.Model.Normalized)
	}

	if v.Count <= 0 || v.Dim <= 0 {
		return fail("vectors are %d x %d", v.Count, v.Dim)
	}
	if v.Dim != m.Model.Dim {
		return fail("vectors.dim %d but model.dim %d", v.Dim, m.Model.Dim)
	}
	if v.Count != m.Ads.Count {
		return fail("%d vectors but %d ads -- rows and ads must correspond", v.Count, m.Ads.Count)
	}
	if want := int64(v.Count) * int64(v.Dim) * 4; v.Bytes != want {
		return fail("vectors.bytes %d, but %d x %d float32 is %d", v.Bytes, v.Count, v.Dim, want)
	}

	// The file names are joined onto the bundle directory, so they must be
	// plain names -- never a path that escapes it.
	for _, name := range []string{v.File, m.Ads.File} {
		if !isPlainName(name) {
			return fail("file name %q is not a plain file name", name)
		}
	}
	if v.SHA256 == "" || m.Ads.SHA256 == "" {
		return fail("missing checksum")
	}
	return nil
}
