/*
 * mind-mem scoring kernels — C99 reference implementations.
 *
 * These are the C equivalents of the MIND tensor kernels in mind/*.mind.
 * When mindc gains --emit-obj in the full build, this file becomes redundant.
 * Until then, gcc -O3 -march=native produces competitive native code.
 *
 * ABI: all functions use flat float arrays + int lengths.
 * Build: gcc -O3 -march=native -shared -fPIC -o libmindmem.so kernels.c -lm
 */

#include <math.h>
#include <stdint.h>

/* ── RRF Fusion ─────────────────────────────────────────────── */

void rrf_fuse(const float *bm25_ranks, const float *vec_ranks,
              int n, float k, float bm25_w, float vec_w,
              float *out) {
    for (int i = 0; i < n; i++)
        out[i] = bm25_w / (k + bm25_ranks[i]) + vec_w / (k + vec_ranks[i]);
}

/* ── BM25F Scoring ──────────────────────────────────────────── */

float bm25f_doc(float tf, float df, float N, float dl, float avgdl,
                float k1, float b, float field_weight) {
    float idf = logf((N - df + 0.5f) / (df + 0.5f) + 1.0f);
    float tf_norm = (tf * (k1 + 1.0f)) / (tf + k1 * (1.0f - b + b * dl / avgdl));
    return idf * tf_norm * field_weight;
}

void bm25f_batch(const float *tfs, float df, float N,
                 const float *dls, float avgdl,
                 float k1, float b, float field_weight,
                 int n, float *out) {
    float idf = logf((N - df + 0.5f) / (df + 0.5f) + 1.0f);
    for (int i = 0; i < n; i++) {
        float tf_norm = (tfs[i] * (k1 + 1.0f)) /
                        (tfs[i] + k1 * (1.0f - b + b * dls[i] / avgdl));
        out[i] = idf * tf_norm * field_weight;
    }
}

/* ── Negation Penalty ───────────────────────────────────────── */

void negation_penalty(const float *scores, const float *neg_flags,
                      float penalty, int n, float *out) {
    for (int i = 0; i < n; i++)
        out[i] = neg_flags[i] > 0.5f ? scores[i] * penalty : scores[i];
}

/* ── Date Proximity (Gaussian decay) ────────────────────────── */

void date_proximity(const float *days_diff, float sigma, int n,
                    float *out) {
    float inv_2sig2 = -0.5f / (sigma * sigma);
    for (int i = 0; i < n; i++)
        out[i] = expf(inv_2sig2 * days_diff[i] * days_diff[i]);
}

/* ── Category Boost ─────────────────────────────────────────── */

void category_boost(const float *scores, const float *match_flags,
                    float boost, int n, float *out) {
    for (int i = 0; i < n; i++)
        out[i] = match_flags[i] > 0.5f ? scores[i] * boost : scores[i];
}

/* ── Importance Scoring (A-MEM) ─────────────────────────────── */

float importance_score_single(int access_count, float days_since,
                              float base_importance, float decay) {
    float recency = expf(-decay * days_since);
    float freq = logf(1.0f + (float)access_count);
    float raw = base_importance * (0.5f + 0.3f * freq + 0.2f * recency);
    if (raw < 0.8f) raw = 0.8f;
    if (raw > 1.5f) raw = 1.5f;
    return raw;
}

void importance_batch(const int *access_counts, const float *days_since,
                      float base, float decay, int n, float *out) {
    for (int i = 0; i < n; i++)
        out[i] = importance_score_single(access_counts[i], days_since[i],
                                         base, decay);
}

/* ── Entity Overlap ─────────────────────────────────────────── */
/* Note: entity overlap involves set operations which don't map to flat arrays.
 * This C version takes pre-computed binary vectors (1.0/0.0) of length E. */

void entity_overlap(const float *query_vec, const float *doc_vecs,
                    int n_docs, int n_entities, float *out) {
    float q_count = 0.0f;
    for (int e = 0; e < n_entities; e++)
        q_count += query_vec[e];
    if (q_count < 0.5f) {
        for (int i = 0; i < n_docs; i++) out[i] = 0.0f;
        return;
    }
    for (int i = 0; i < n_docs; i++) {
        float overlap = 0.0f;
        const float *doc = doc_vecs + i * n_entities;
        for (int e = 0; e < n_entities; e++)
            overlap += query_vec[e] * doc[e];
        out[i] = overlap / q_count;
    }
}

/* ── Confidence Score ───────────────────────────────────────── */

float confidence_score(float entity_ov, float bm25_norm,
                       float speaker_cov, float evidence_density,
                       float negation_asym,
                       float w0, float w1, float w2, float w3, float w4) {
    return entity_ov * w0 + bm25_norm * w1 + speaker_cov * w2 +
           evidence_density * w3 + negation_asym * w4;
}

/* ── Top-K Mask ─────────────────────────────────────────────── */
/* Simple selection sort for top-k. Could use nth_element for large N. */

void top_k_mask(const float *scores, int n, int k, float *out) {
    /* Initialize all to 0 */
    for (int i = 0; i < n; i++) out[i] = 0.0f;

    if (k >= n) {
        for (int i = 0; i < n; i++) out[i] = 1.0f;
        return;
    }

    /* Find k-th largest via partial sort (copy + partition) */
    /* For benchmark sizes, a simple approach suffices */
    for (int found = 0; found < k; found++) {
        int best_idx = -1;
        float best_val = -1e30f;
        for (int i = 0; i < n; i++) {
            if (out[i] < 0.5f && scores[i] > best_val) {
                best_val = scores[i];
                best_idx = i;
            }
        }
        if (best_idx >= 0) out[best_idx] = 1.0f;
    }
}

/* ── Weighted Rank ──────────────────────────────────────────── */

void weighted_rank(const float *scores, const float *weights,
                   int n, float *out) {
    for (int i = 0; i < n; i++)
        out[i] = scores[i] * weights[i];
}

/* ── Category Affinity (distillation scoring) ──────────────── */
/* Weighted sum of keyword/tag/entity overlap matrices [N*C flat]. */

void category_affinity(const float *kw_overlap, const float *tag_match,
                       const float *ent_match,
                       float kw_w, float tag_w, float ent_w,
                       int n_blocks, int n_cats, float *out) {
    int total = n_blocks * n_cats;
    for (int i = 0; i < total; i++)
        out[i] = kw_w * kw_overlap[i] + tag_w * tag_match[i] + ent_w * ent_match[i];
}

/* ── Query-Category Relevance ──────────────────────────────── */
/* Dot product of query keyword vector [K] against category profiles [C*K]. */

void query_category_relevance(const float *query_kw, const float *cat_kw,
                               int n_cats, int n_keywords, float *out) {
    float q_sum = 0.0f;
    for (int k = 0; k < n_keywords; k++)
        q_sum += query_kw[k];
    if (q_sum < 1e-6f) q_sum = 1e-6f;

    for (int c = 0; c < n_cats; c++) {
        float dot = 0.0f;
        const float *row = cat_kw + c * n_keywords;
        for (int k = 0; k < n_keywords; k++)
            dot += row[k] * query_kw[k];
        out[c] = dot / q_sum;
    }
}

/* ── Category Assignment (soft sigmoid threshold) ──────────── */
/* Steep sigmoid around threshold. Steepness 20 gives ~0.01 at -0.23. */

void category_assign(const float *affinity, float threshold,
                     int n_blocks, int n_cats, float *out) {
    int total = n_blocks * n_cats;
    for (int i = 0; i < total; i++) {
        float shifted = (affinity[i] - threshold) * 20.0f;
        out[i] = 1.0f / (1.0f + expf(-shifted));
    }
}

/* ── Runtime Protection Probe ──────────────────────────────── */
/* Returns 1 if FORTRESS protection is linked, 0 for dev/CI builds.
 * Protected builds replace this with the real check from protection.mind. */

int mindmem_protected(void) {
#ifdef MIND_PROTECTED
    return 1;
#else
    return 0;
#endif
}
