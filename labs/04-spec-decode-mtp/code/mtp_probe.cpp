/**
 * MTP Probe — MoE MTP accept/reject side-channel experiment harness
 * ====================================================================
 * Based on llama.cpp speculative-simple example.
 * Modified to support MTP (LLAMA_CONTEXT_TYPE_MTP) + per-step logging.
 *
 * Build:
 *   cd llama.cpp && cmake --build build -j4
 *   g++ -std=c++17 -O2 -I include -I common -I build/tools/cli
 *       experiments/mtp_probe.cpp -o build/bin/mtp_probe
 *       -L build/bin -llama -lggml -lcommon -lcommon-impl -lspeculative-impl
 *
 * Usage:
 *   ./build/bin/mtp_probe \
 *       -m model.gguf --gpu-layers 99 \
 *       -p "prompt1" "prompt2" ... \
 *       --spec-type draft-mtp --spec-draft-n-max 2 \
 *       -n 32 --temp 0.0
 */

#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <string>
#include <vector>
#include <utility>

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    // Enable MTP if spec_type is draft-mtp
    const bool use_mtp = std::find(params.speculative.types.begin(),
                                   params.speculative.types.end(),
                                   COMMON_SPECULATIVE_TYPE_DRAFT_MTP)
                         != params.speculative.types.end();

    // Collect prompts from remaining args or use default
    std::vector<std::string> prompts;
    if (params.prompt.empty()) {
        prompts.push_back("The capital of France is");
    } else {
        prompts.push_back(params.prompt);
    }

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    // load the target model
    auto llama_init_tgt = common_init_from_params(params);

    llama_model * model_tgt = llama_init_tgt->model();
    llama_context * ctx_tgt  = llama_init_tgt->context();

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    // create draft context (MTP shares the target model)
    llama_model_ptr model_dft;
    llama_context_ptr ctx_dft;

    if (use_mtp) {
        // MTP: draft context from same model, special context type
        auto cparams_dft = common_context_params_to_llama(params);
        cparams_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        ctx_dft.reset(llama_init_from_model(model_tgt, cparams_dft));
        if (!ctx_dft) {
            LOG_ERR("failed to create MTP draft context\n");
            return 1;
        }

        int n_mtp_layers = std::max(1, (int)llama_model_n_layer_nextn(model_tgt));
        LOG_INF("MTP enabled: %d nextn layer(s)\n", n_mtp_layers);

        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = ctx_dft.get();
    } else {
        LOG_ERR("This tool requires --spec-type draft-mtp\n");
        return 1;
    }

    // === Process each prompt ===

    for (size_t pi = 0; pi < prompts.size(); pi++) {
        const std::string & prompt = prompts[pi];

        LOG_INF("\n=== Prompt %zu/%zu: \"%s\" ===\n", pi + 1, prompts.size(), prompt.c_str());

        // Tokenize
        std::vector<llama_token> inp;
        inp = common_tokenize(ctx_tgt, prompt, true, true);

        if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) {
            LOG_ERR("prompt exceeds batch size\n");
            return 1;
        }

        // Per-step accept/reject log for this prompt
        struct StepRecord {
            int step;
            int n_draft;
            int n_accept;
            double t0_us;
            double decode_us;
        };
        std::vector<StepRecord> steps;

        // MoE expert activation counters: [layer][expert_idx] = count
    // Aggregate across all decode steps for this prompt
    std::vector<std::vector<int64_t>> expert_activation_count;
    std::vector<std::vector<int64_t>> expert_accept;
    std::vector<std::vector<int64_t>> expert_reject;

        int n_predict = 0;
        int n_drafted_total = 0;
        int n_accept_total  = 0;
        bool has_eos = false;

        llama_seq_id seq_id = 0;

        common_sampler_ptr smpl(common_sampler_init(model_tgt, params.sampling));

        // Eval prompt on both contexts
        llama_decode(ctx_tgt,  llama_batch_get_one(inp.data(), inp.size() - 1));
        llama_decode(ctx_dft.get(), llama_batch_get_one(inp.data(), inp.size() - 1));

        llama_token id_last = inp.back();

        llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
        prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

        int n_past = inp.size() - 1;

        // Init speculator
        struct common_speculative * spec = common_speculative_init(params.speculative, 1);
        common_speculative_begin(spec, seq_id, prompt_tgt);

        llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

        size_t n_draft = 0;
        llama_tokens draft;

        // --- Main decode loop ---
        const auto t_start = ggml_time_us();

        // Capture expert routing — aggregate across all decode steps
        int n_moe_layers = 0;
        int n_moe_tokens = 0;
        int n_exp_used   = 0;
        int n_expert_total = 0;

        // Temporary storage: expert indices from the most recent decode, for per-expert accept attribution
        std::vector<std::vector<int32_t>> saved_expert_indices; // [layer][n_exp_used * n_tokens]
        int saved_n_exp = 0, saved_n_tok = 0;

        bool first_expert_capture_done = false;
        while (true) {
            double t_step_start = ggml_time_us();

            if (draft.empty()) {
                // generate new draft
                common_speculative_get_draft_params(spec, seq_id) = {
                    /* .drafting   = */ true,
                    /* .n_max      = */ -1,
                    /* .n_past     = */ n_past,
                    /* .id_last    = */ id_last,
                    /* .prompt     = */ &prompt_tgt,
                    /* .result     = */ &draft,
                };
                common_speculative_draft(spec);

                n_draft = draft.size();

                // prep target batch: [id_last, draft0, draft1, ...]
                common_batch_clear(batch_tgt);
                common_batch_add(batch_tgt, id_last, n_past++, {seq_id}, true);

                for (size_t i = 0; i < draft.size(); ++i) {
                    common_batch_add(batch_tgt, draft[i], n_past + i, {seq_id}, true);
                }

                llama_decode(ctx_tgt, batch_tgt);

                // Save expert indices right after decode (before they get overwritten)
                saved_expert_indices.clear();
                saved_n_exp = llama_get_moe_n_expert_used(ctx_tgt);
                saved_n_tok = llama_get_moe_n_tokens(ctx_tgt);
                int saved_n_layers = llama_get_moe_n_layers(ctx_tgt);
                for (int l = 0; l < saved_n_layers; l++) {
                    const int32_t * d = llama_get_moe_expert_indices(ctx_tgt, l);
                    if (d && saved_n_exp > 0 && saved_n_tok > 0) {
                        saved_expert_indices.push_back(
                            std::vector<int32_t>(d, d + saved_n_exp * saved_n_tok));
                    } else {
                        saved_expert_indices.push_back({});
                    }
                }

                // Capture MoE data right after decode — read raw tensor data
                {
                    n_moe_layers = llama_get_moe_n_layers(ctx_tgt);
                    n_moe_tokens = llama_get_moe_n_tokens(ctx_tgt);
                    n_exp_used   = llama_get_moe_n_expert_used(ctx_tgt);

                    if (expert_activation_count.empty() && n_moe_layers > 0) {
                        n_expert_total = 256;
                        expert_activation_count.resize(n_moe_layers,
                            std::vector<int64_t>(n_expert_total, 0));
                        expert_accept.resize(n_moe_layers,
                            std::vector<int64_t>(n_expert_total, 0));
                        expert_reject.resize(n_moe_layers,
                            std::vector<int64_t>(n_expert_total, 0));
                    }

                    for (int l = 0; l < n_moe_layers && l < (int)expert_activation_count.size(); l++) {
                        const int32_t * exp_data = llama_get_moe_expert_indices(ctx_tgt, l);
                        if (exp_data && n_exp_used > 0 && n_moe_tokens > 0) {
                            for (int t = 0; t < n_moe_tokens; t++) {
                                for (int e = 0; e < n_exp_used; e++) {
                                    int32_t expert_idx = exp_data[t * n_exp_used + e];
                                    if (expert_idx >= 0 && expert_idx < n_expert_total) {
                                        expert_activation_count[l][expert_idx]++;
                                    }
                                }
                            }
                        }
                    }

                    // Capture valid expert data from FIRST decode only
                    if (!first_expert_capture_done && n_moe_layers > 0 && n_moe_tokens > 0) {
                        fprintf(stderr, "MOE_RAW n_layers=%d n_exp=%d n_tok=%d\n",
                            n_moe_layers, n_exp_used, n_moe_tokens);
                        for (int l = 0; l < n_moe_layers; l++) {
                            const int32_t * d = llama_get_moe_expert_indices(ctx_tgt, l);
                            if (d) {
                                fprintf(stderr, "MOE_L%d", l);
                                for (int i = 0; i < n_exp_used * n_moe_tokens; i++) {
                                    fprintf(stderr, " %d", d[i]);
                                }
                                fprintf(stderr, "\n");
                            }
                        }
                        first_expert_capture_done = true;
                    }
                }
            }

            // Sample and get accepted tokens
            auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);

            if (ids.empty()) break;

            int n_accepted = (int)ids.size() - 1;  // minus the verified token

            // Per-expert MTP accept/reject attribution (use saved copy from decode)
            if (!saved_expert_indices.empty() && saved_n_exp > 0 && saved_n_tok > 1) {
                for (int l = 0; l < (int)saved_expert_indices.size() && l < (int)expert_accept.size(); l++) {
                    const auto & exp_vec = saved_expert_indices[l];
                    if (exp_vec.empty()) continue;
                    // Draft token positions: 1..saved_n_tok-1
                    for (int pos = 1; pos < saved_n_tok; pos++) {
                        bool accepted = (pos - 1) < n_accepted;
                        auto & counter = accepted ? expert_accept[l] : expert_reject[l];
                        for (int e = 0; e < saved_n_exp; e++) {
                            int idx = pos * saved_n_exp + e;
                            if (idx < (int)exp_vec.size()) {
                                int32_t expert_idx = exp_vec[idx];
                                if (expert_idx >= 0 && expert_idx < (int)counter.size()) {
                                    counter[expert_idx]++;
                                }
                            }
                        }
                    }
                }
            }

            StepRecord rec;
            rec.step        = (int)steps.size();
            rec.n_draft     = (int)n_draft;
            rec.n_accept    = n_accepted;
            rec.t0_us       = t_step_start;
            rec.decode_us   = ggml_time_us() - t_step_start;
            steps.push_back(rec);

            n_drafted_total += (int)n_draft;
            n_accept_total  += n_accepted;
            n_predict       += (int)ids.size();

            // Process accepted tokens
            for (size_t i = 0; i < ids.size(); ++i) {
                prompt_tgt.push_back(id_last);
                id_last = ids[i];

                if (llama_vocab_is_eog(vocab, id_last)) {
                    has_eos = true;
                    break;
                }

                std::string token_str = common_token_to_piece(ctx_tgt, id_last);
                LOG("%s", token_str.c_str());
            }

            draft.clear();

            // cleanup
            llama_memory_seq_rm(llama_get_memory(ctx_tgt),       seq_id, n_past, -1);
            llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), seq_id, n_past, -1);

            if ((params.n_predict >= 0 && n_predict >= params.n_predict) || has_eos) {
                break;
            }
        }

        auto t_end = ggml_time_us();

        LOG("\n");
        LOG_INF("--- Prompt %zu stats ---\n", pi + 1);
        LOG_INF("n_draft_max = %d\n", params.speculative.draft.n_max);
        LOG_INF("n_predict   = %d\n", n_predict);
        LOG_INF("n_drafted   = %d\n", n_drafted_total);
        LOG_INF("n_accept    = %d\n", n_accept_total);
        LOG_INF("accept_rate = %.1f%%\n",
                n_drafted_total > 0 ? 100.0 * n_accept_total / n_drafted_total : 0.0);
        LOG_INF("total_time  = %.3f s\n", (t_end - t_start) / 1e6);
        LOG_INF("speed       = %.1f t/s\n", n_predict / ((t_end - t_start) / 1e6));

        // Print per-step CSV for analysis
        LOG_INF("\n--- Per-step log (CSV) ---\n");
        LOG_INF("step,n_draft,n_accept,accept_rate,time_us\n");
        for (const auto & s : steps) {
            double ar = s.n_draft > 0 ? 100.0 * s.n_accept / s.n_draft : 0.0;
            LOG_INF("%d,%d,%d,%.1f%%,%.0f\n", s.step, s.n_draft, s.n_accept, ar, s.decode_us);
        }

        // Print per-expert MTP accept rate
        if (!expert_accept.empty() && !expert_reject.empty()) {
            fprintf(stderr, "\n--- Per-Expert MTP Accept Rate ---\n");
            fprintf(stderr, "PER_EXPERT_ACCEPT n_layers=%d n_experts=%d\n",
                (int)expert_accept.size(), n_expert_total);
            for (int l = 0; l < (int)expert_accept.size(); l++) {
                fprintf(stderr, "PEA_L%d", l);
                for (int e = 0; e < n_expert_total; e++) {
                    int64_t acc = expert_accept[l][e];
                    int64_t rej = expert_reject[l][e];
                    int64_t total = acc + rej;
                    if (total > 0) {
                        // Print expert_id,accept_rate,accept_count,reject_count
                        double rate = 100.0 * acc / total;
                        fprintf(stderr, " e%d:%.1f:%lld:%lld", e, rate, (long long)acc, (long long)rej);
                    }
                }
                fprintf(stderr, "\n");
            }
        }

        // Print MoE expert routing summary (aggregated across all decode steps)
        LOG_INF("\n--- MoE Expert Routing (aggregated) ---\n");
        LOG_INF("n_moe_layers = %d\n", (int)expert_activation_count.size());
        LOG_INF("n_exp_used   = %d (per token)\n", n_exp_used);
        LOG_INF("n_expert_total = %d\n", n_expert_total);

        if (!expert_activation_count.empty()) {
            // Print first 10 layers: top-10 most-used experts
            int n_show = std::min(10, (int)expert_activation_count.size());
            for (int l = 0; l < n_show; l++) {
                const auto & counts = expert_activation_count[l];
                // Find top-10 experts by activation count
                std::vector<std::pair<int64_t, int>> sorted;
                for (int e = 0; e < (int)counts.size(); e++) {
                    if (counts[e] > 0) sorted.push_back({counts[e], e});
                }
                std::sort(sorted.rbegin(), sorted.rend());
                LOG_INF("Layer %d: ", l);
                for (int i = 0; i < std::min(10, (int)sorted.size()); i++) {
                    LOG("e%d:%lld ", sorted[i].second, (long long)sorted[i].first);
                }
                LOG("\n");
            }
            // CSV: per-layer expert activation distribution for the first layer
            LOG_INF("\nLayer 0 expert activation CSV (expert_id,count):\n");
            const auto & l0 = expert_activation_count[0];
            for (int e = 0; e < (int)l0.size(); e++) {
                if (l0[e] > 0) {
                    LOG("%d,%lld\n", e, (long long)l0[e]);
                }
            }
        } else {
            LOG_INF("(no expert routing data captured)\n");
        }

        common_speculative_free(spec);
        llama_batch_free(batch_tgt);
    }

    llama_backend_free();

    LOG("\n");
    return 0;
}
