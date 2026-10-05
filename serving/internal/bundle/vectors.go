package bundle

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"math"
	"os"
)

// NormTolerance matches NORM_TOLERANCE in ingestion/publish.py. Floating
// point makes "unit length" mean 1.0 give or take; an all-zero row (norm 0)
// is the failure this is really hunting for.
const NormTolerance = 1e-3

// readChunkBytes is how much of vectors.f32 is held as raw bytes at once. It
// must be a multiple of 4 so a float32 never straddles two chunks.
const readChunkBytes = 1 << 20

// readVectors reads vectors.f32 into one flat []float32, hashing the bytes as
// they go past, then checks every row.
//
// The file is read in fixed-size chunks and decoded straight into the final
// slice, so peak memory is the vectors plus 1MB -- not the vectors twice, as
// reading the whole file into a []byte first would cost.
func readVectors(path string, spec VectorsSpec) ([]float32, error) {
	// Size first: it catches a truncated or padded file without reading a
	// single byte of it.
	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrVectors, err)
	}
	if info.Size() != spec.Bytes {
		return nil, fmt.Errorf("%w: %s is %d bytes, manifest says %d",
			ErrVectors, path, info.Size(), spec.Bytes)
	}

	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrVectors, err)
	}
	defer f.Close()

	// One flat slice rather than [][]float32: a single allocation, contiguous
	// in memory, so a scan over every row walks memory in order. Row i is
	// vectors[i*dim : (i+1)*dim].
	vectors := make([]float32, spec.Count*spec.Dim)
	hasher := sha256.New()
	buf := make([]byte, readChunkBytes)

	next := 0
	for next < len(vectors) {
		want := min(len(buf), (len(vectors)-next)*4)
		n, err := io.ReadFull(f, buf[:want])
		if err != nil {
			return nil, fmt.Errorf("%w: reading %s: %v", ErrVectors, path, err)
		}
		hasher.Write(buf[:n])
		// Decoded explicitly as little-endian rather than reinterpreting the
		// bytes with unsafe: correct on any CPU, and it costs milliseconds.
		for off := 0; off < n; off += 4 {
			vectors[next] = math.Float32frombits(binary.LittleEndian.Uint32(buf[off:]))
			next++
		}
	}

	if got := hex.EncodeToString(hasher.Sum(nil)); got != spec.SHA256 {
		return nil, fmt.Errorf("%w: %s sha256 %.12s, manifest says %.12s",
			ErrChecksum, path, got, spec.SHA256)
	}

	if err := checkRows(vectors, spec.Dim); err != nil {
		return nil, err
	}
	return vectors, nil
}

// checkRows refuses numbers that cannot mean anything. publish.py already
// checks this, and the checksum proves the bytes are the ones it wrote -- but
// the loader is the last line of defence before serving, so it does not take
// another program's word for it.
//
// A NaN compares false against everything, so its row is invisible to search.
// A zero row scores 0 against every query and can never be retrieved.
func checkRows(vectors []float32, dim int) error {
	for row := 0; row*dim < len(vectors); row++ {
		var sumSq float64
		for _, x := range vectors[row*dim : (row+1)*dim] {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				return fmt.Errorf("%w: row %d contains NaN or infinity", ErrVectors, row)
			}
			sumSq += float64(x) * float64(x)
		}
		if norm := math.Sqrt(sumSq); math.Abs(norm-1) > NormTolerance {
			return fmt.Errorf("%w: row %d has norm %.6f, want 1.0 (dot product is only cosine "+
				"similarity for unit vectors)", ErrVectors, row, norm)
		}
	}
	return nil
}
