package bundle

import (
	"container/heap"
	"fmt"
	"sort"
)

// Hit is one search result: a vector row and its similarity to the query.
type Hit struct {
	Row   int
	Score float32
}

// TopK returns the k rows most similar to query, best first.
//
// It is EXACT: the query is scored against every row. That makes it slow
// relative to an approximate index, and that is the point -- it is the ground
// truth an approximate index's recall is measured against.
//
// The vectors are unit length, so the dot product IS cosine similarity. No
// square roots, no normalising.
func (b *Bundle) TopK(query []float32, k int) ([]Hit, error) {
	if len(query) != b.Dim {
		return nil, fmt.Errorf("query has %d dims, bundle has %d", len(query), b.Dim)
	}
	k = min(k, b.Len())
	if k <= 0 {
		return nil, nil
	}

	// A min-heap of the best k seen so far, with the WEAKEST of them on top.
	// Each new row only has to beat that one to get in, so the work is
	// O(N log k) instead of sorting all N scores.
	h := make(minHeap, 0, k)
	for row := 0; row < b.Len(); row++ {
		hit := Hit{Row: row, Score: dot(query, b.Vector(row))}
		if len(h) < k {
			heap.Push(&h, hit)
		} else if worse(h[0], hit) {
			h[0] = hit
			heap.Fix(&h, 0)
		}
	}

	hits := []Hit(h)
	sort.Slice(hits, func(i, j int) bool { return worse(hits[j], hits[i]) })
	return hits, nil
}

func dot(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// worse reports whether a ranks below b. Ties on score go to the lower row,
// so results are deterministic run to run.
func worse(a, b Hit) bool {
	if a.Score != b.Score {
		return a.Score < b.Score
	}
	return a.Row > b.Row
}

// minHeap implements container/heap with the worst hit at index 0.
type minHeap []Hit

func (h minHeap) Len() int           { return len(h) }
func (h minHeap) Less(i, j int) bool { return worse(h[i], h[j]) }
func (h minHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x any)        { *h = append(*h, x.(Hit)) }
func (h *minHeap) Pop() any {
	old := *h
	last := old[len(old)-1]
	*h = old[:len(old)-1]
	return last
}
