package bundle

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadHappyPath(t *testing.T) {
	f := newFixture(20, 8)
	b, err := Load(f.write(t))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if b.Version != fixtureVersion {
		t.Errorf("Version = %q, want %q", b.Version, fixtureVersion)
	}
	if b.Len() != 20 || b.Dim != 8 {
		t.Fatalf("loaded %d x %d, want 20 x 8", b.Len(), b.Dim)
	}
	// Every value must come back bit-for-bit: this is what proves the
	// little-endian decode and the row offsets are right.
	for row := 0; row < b.Len(); row++ {
		got, want := b.Vector(row), f.vectors[row*8:(row+1)*8]
		for j := range want {
			if math.Float32bits(got[j]) != math.Float32bits(want[j]) {
				t.Fatalf("row %d dim %d = %v, want %v", row, j, got[j], want[j])
			}
		}
		if b.Ads[row].Row != row {
			t.Errorf("Ads[%d].Row = %d", row, b.Ads[row].Row)
		}
		if b.RowByID[b.Ads[row].AdID] != row {
			t.Errorf("RowByID[%s] = %d, want %d", b.Ads[row].AdID, b.RowByID[b.Ads[row].AdID], row)
		}
	}
}

// Each case damages a valid bundle in exactly one way and asserts that Load
// refuses it with the right kind of error. The loader is only trustworthy if
// every one of these has been seen to fire.
func TestLoadRejects(t *testing.T) {
	cases := []struct {
		name    string
		damage  func(f *fixture)
		wantErr error
	}{
		// --- LATEST ---------------------------------------------------------
		{"missing LATEST", func(f *fixture) {
			f.after = func(t *testing.T, root, _ string) { removeFile(t, filepath.Join(root, LatestPointer)) }
		}, ErrManifest},
		{"empty LATEST", func(f *fixture) {
			f.after = func(t *testing.T, root, _ string) { writeFile(t, filepath.Join(root, LatestPointer), []byte("  \n")) }
		}, ErrManifest},
		{"LATEST escapes root", func(f *fixture) {
			f.after = func(t *testing.T, root, _ string) { writeFile(t, filepath.Join(root, LatestPointer), []byte("../elsewhere\n")) }
		}, ErrManifest},

		// --- manifest contract ----------------------------------------------
		{"unknown format version", func(f *fixture) {
			f.editManifest = func(m *Manifest) { m.FormatVersion = 2 }
		}, ErrManifest},
		{"big-endian vectors", func(f *fixture) {
			f.editManifest = func(m *Manifest) { m.Vectors.ByteOrder = "big" }
		}, ErrManifest},
		{"not normalised", func(f *fixture) {
			f.editManifest = func(m *Manifest) { m.Model.Normalized = false }
		}, ErrManifest},
		{"vector dim differs from model dim", func(f *fixture) {
			f.editManifest = func(m *Manifest) { m.Model.Dim = 16 }
		}, ErrManifest},
		{"vector count differs from ad count", func(f *fixture) {
			f.editManifest = func(m *Manifest) { m.Ads.Count++ }
		}, ErrManifest},
		{"byte count disagrees with shape", func(f *fixture) {
			f.editManifest = func(m *Manifest) { m.Vectors.Bytes += 4 }
		}, ErrManifest},
		{"file name escapes bundle", func(f *fixture) {
			f.editManifest = func(m *Manifest) { m.Vectors.File = "../vectors.f32" }
		}, ErrManifest},
		{"manifest is not JSON", func(f *fixture) {
			f.after = func(t *testing.T, _, dir string) { writeFile(t, filepath.Join(dir, "manifest.json"), []byte("{")) }
		}, ErrManifest},

		// --- vectors ----------------------------------------------------------
		{"truncated vectors file", func(f *fixture) {
			f.after = func(t *testing.T, _, dir string) {
				path := filepath.Join(dir, "vectors.f32")
				if err := os.Truncate(path, fileSize(t, path)-4); err != nil {
					t.Fatal(err)
				}
			}
		}, ErrVectors},
		{"one flipped byte in vectors", func(f *fixture) {
			f.after = func(t *testing.T, _, dir string) { flipByte(t, filepath.Join(dir, "vectors.f32"), 37) }
		}, ErrChecksum},
		{"NaN in a row", func(f *fixture) {
			f.vectors[5*f.dim+2] = float32(math.NaN())
		}, ErrVectors},
		{"infinity in a row", func(f *fixture) {
			f.vectors[7*f.dim] = float32(math.Inf(1))
		}, ErrVectors},
		{"all-zero row", func(f *fixture) {
			f.setRow(3, 0)
		}, ErrVectors},
		{"row not unit length", func(f *fixture) {
			f.setRow(4, 0.5) // norm = 0.5 * sqrt(8) ≈ 1.41
		}, ErrVectors},

		// --- ads --------------------------------------------------------------
		{"ads out of order", func(f *fixture) {
			f.adLines[2], f.adLines[3] = f.adLines[3], f.adLines[2]
		}, ErrAds},
		{"ad without a row field", func(f *fixture) {
			f.adLines[0] = `{"ad_id": "ad_0000", "headline": "no row field here"}`
		}, ErrAds},
		{"duplicate ad_id", func(f *fixture) {
			f.adLines[4] = adLine(4, "ad_0001")
		}, ErrAds},
		{"ad without an ad_id", func(f *fixture) {
			f.adLines[6] = adLine(6, "")
		}, ErrAds},
		{"fewer ads than vectors", func(f *fixture) {
			f.adLines = f.adLines[:len(f.adLines)-1]
		}, ErrAds},
		{"malformed ad line", func(f *fixture) {
			f.adLines[9] = `{"row": 9, "ad_id": `
		}, ErrAds},
		{"ads checksum mismatch", func(f *fixture) {
			f.editManifest = func(m *Manifest) { m.Ads.SHA256 = sha([]byte("something else")) }
		}, ErrChecksum},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := newFixture(20, 8)
			tc.damage(f)
			b, err := Load(f.write(t))
			if err == nil {
				t.Fatalf("Load succeeded (%d ads), want %v", b.Len(), tc.wantErr)
			}
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("Load error = %v, want %v", err, tc.wantErr)
			}
		})
	}
}

// A line longer than bufio.Scanner's 64KB default must still load. Without
// the bigger buffer, Scan stops early with an error that is easy to miss.
func TestLoadLongAdLine(t *testing.T) {
	f := newFixture(3, 8)
	long, _ := json.Marshal(Ad{Row: 1, AdID: "ad_long", Description: strings.Repeat("x", 100_000)})
	f.adLines[1] = string(long)
	if _, err := Load(f.write(t)); err != nil {
		t.Fatalf("Load: %v", err)
	}
}

func removeFile(t *testing.T, path string) {
	t.Helper()
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
}

func fileSize(t *testing.T, path string) int64 {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	return info.Size()
}

func flipByte(t *testing.T, path string, offset int) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	data[offset] ^= 0x01
	writeFile(t, path, data)
}
