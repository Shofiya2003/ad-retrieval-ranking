// Command loadcheck loads and verifies the latest published bundle, then
// reports what it found. It is the Go side's equivalent of the Python stage
// gates: a non-zero exit means the bundle must not be served.
//
//	go run ./cmd/loadcheck -root ../data/published
package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"runtime"
	"time"

	"github.com/Shofiya2003/ad-retrieval-ranking/serving/internal/bundle"
)

// Exit codes match the Python stages.
const (
	exitOK           = 0
	exitGateFailed   = 1
	exitInputMissing = 2
)

func main() {
	root := flag.String("root", "data/published", "directory holding LATEST and the bundle versions")
	flag.Parse()
	os.Exit(run(*root))
}

func run(root string) int {
	runtime.GC()
	var before runtime.MemStats
	runtime.ReadMemStats(&before)

	started := time.Now()
	b, err := bundle.Load(root)
	elapsed := time.Since(started)

	if err != nil {
		fmt.Fprintf(os.Stderr, "LOAD REFUSED: %v\n", err)
		if errors.Is(err, os.ErrNotExist) {
			return exitInputMissing
		}
		return exitGateFailed
	}

	// Heap in use after a GC, so it counts what the bundle keeps, not the
	// garbage left over from reading it.
	runtime.GC()
	var after runtime.MemStats
	runtime.ReadMemStats(&after)
	heapMB := float64(int64(after.HeapAlloc)-int64(before.HeapAlloc)) / (1 << 20)

	m := b.Manifest
	rule := "=============================================================="
	fmt.Println()
	fmt.Println(rule)
	fmt.Println("  LOADED BUNDLE")
	fmt.Println(rule)
	fmt.Printf("  version    %s\n", b.Version)
	fmt.Printf("  model      %s @ %.12s\n", m.Model.Name, m.Model.Revision)
	fmt.Printf("  vectors    %s x %d float32   checksum ok   norms ok\n", commas(m.Vectors.Count), b.Dim)
	fmt.Printf("  ads        %s   checksum ok   rows aligned   ids unique\n", commas(b.Len()))
	fmt.Printf("  load       %d ms\n", elapsed.Milliseconds())
	fmt.Printf("  heap       %.1f MB\n", heapMB)
	fmt.Println(rule)
	fmt.Println()

	// Keep the bundle reachable until after the second GC, so it is counted.
	runtime.KeepAlive(b)
	return exitOK
}

func commas(n int) string {
	s := fmt.Sprint(n)
	for i := len(s) - 3; i > 0; i -= 3 {
		s = s[:i] + "," + s[i:]
	}
	return s
}
