package bundle

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// maxLineBytes caps one ads.jsonl line. bufio.Scanner's default is 64KB, and a
// line longer than that is not "skipped" -- Scan just stops with an error that
// is easy to miss. Validation caps text far below this, so 1MB is generous.
const maxLineBytes = 1 << 20

// Ad is one line of ads.jsonl. Row is the index of its vector.
type Ad struct {
	Row          int     `json:"row"`
	AdID         string  `json:"ad_id"`
	Headline     string  `json:"headline"`
	Description  string  `json:"description"`
	Category     string  `json:"category"`
	AdvertiserID string  `json:"advertiser_id"`
	BidCPMUSD    float64 `json:"bid_cpm_usd"`
	CreatedAt    string  `json:"created_at"`
}

// readAds streams ads.jsonl, hashing it on the way through, and proves that
// line i is the ad for vector row i.
//
// That alignment is the one thing that cannot be allowed to drift: if it
// does, nothing crashes -- every query just returns a different ad from the
// one whose vector matched.
func readAds(path string, spec AdsSpec) ([]Ad, map[string]int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("%w: %w", ErrAds, err)
	}
	defer f.Close()

	hasher := sha256.New()
	scanner := bufio.NewScanner(io.TeeReader(f, hasher))
	scanner.Buffer(make([]byte, 64*1024), maxLineBytes)

	ads := make([]Ad, 0, spec.Count)
	rowByID := make(map[string]int, spec.Count)

	for line := 0; scanner.Scan(); line++ {
		// Row is a pointer only here, so a MISSING row field is caught rather
		// than decoding as 0 and passing for line 0.
		var raw struct {
			Ad
			Row *int `json:"row"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &raw); err != nil {
			return nil, nil, fmt.Errorf("%w: line %d: %v", ErrAds, line, err)
		}
		if raw.Row == nil {
			return nil, nil, fmt.Errorf("%w: line %d has no row field", ErrAds, line)
		}
		if *raw.Row != line {
			return nil, nil, fmt.Errorf("%w: line %d claims row %d -- ads and vectors are misaligned",
				ErrAds, line, *raw.Row)
		}
		if raw.AdID == "" {
			return nil, nil, fmt.Errorf("%w: line %d has no ad_id", ErrAds, line)
		}
		if first, seen := rowByID[raw.AdID]; seen {
			return nil, nil, fmt.Errorf("%w: ad_id %s on rows %d and %d", ErrAds, raw.AdID, first, line)
		}

		ad := raw.Ad
		ad.Row = line
		ads = append(ads, ad)
		rowByID[ad.AdID] = line
	}
	if err := scanner.Err(); err != nil {
		return nil, nil, fmt.Errorf("%w: reading %s: %v", ErrAds, path, err)
	}

	if len(ads) != spec.Count {
		return nil, nil, fmt.Errorf("%w: %s has %d ads, manifest says %d", ErrAds, path, len(ads), spec.Count)
	}
	// The scanner has read to EOF, so every byte has gone through the hasher.
	if got := hex.EncodeToString(hasher.Sum(nil)); got != spec.SHA256 {
		return nil, nil, fmt.Errorf("%w: %s sha256 %.12s, manifest says %.12s",
			ErrChecksum, path, got, spec.SHA256)
	}
	return ads, rowByID, nil
}
