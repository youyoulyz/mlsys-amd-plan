/**
 * MTP Probe (no-instrumentation variant) — 2026-08-09
 * =====================================================
 * 从 mtp_probe.cpp 精简:去除 MoE expert 插桩(llama_get_moe_*),
 * 保留 MTP speculative 测量(draft/accept/逐步时延/CSV)。
 * 用途: ① 在 mtp-clean 分支上编译(无补丁 API 依赖)
 *        ② 无探针对照(验证探针干扰假设)
 *
 * Build (mtp-clean 或 master):
 *   g++ -std=c++17 -O2 -I include -I common -I ggml/include -I build/tools/cli \
 *       experiments/mtp_probe_noprobe.cpp -o build/bin/mtp_probe_noprobe \
 *       -L build/bin -llama -lggml -lcommon -lcommon-impl -lspeculative-impl
 *
 * Usage:
 *   ./build/bin/mtp_probe_noprobe -m model.gguf --gpu-layers 99 \
 *       -p "prompt" --spec-type draft-mtp --spec-draft-n-max 2 -n 64 --temp 0.6
 */

#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    const bool use_mtp = std::find(params.speculative.types.begin(),
                                   params.speculative.types.end(),
                                   COMMON_SPECULATIVE_TYPE_DRAFT_MTP)
                         != params.speculative.types.end();

    std::vector<std::string> prompts;
    if (params.prompt.empty()) {
        prompts.push_back("The capital of France is");
    } else {
        prompts.push_back(params.prompt);
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init_tgt = common_init_from_params(params);
    llama_model * model_tgt = llama_init_tgt->model();
    llama_context * ctx_tgt  = llama_init_tgt->context();
    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    llama_model_ptr model_dft;
    llama_context_ptr ctx_dft;

    if (use_mtp) {
        auto cparams_dft = common_context_params_to_llama(params);
        cparams_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        ctx_dft.reset(llama_init_from_model(model_tgt, cparams_dft));
        if (!ctx_dft) {
            LOG_ERR("failed to create MTP draft context\n");
            return 1;
        }
        LOG_INF("MTP enabled\n");

        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = ctx_dft.get();
    } else {
        LOG_ERR("This tool requires --spec-type draft-mtp\n");
        return 1;
    }

    for (size_t pi = 0; pi < prompts.size(); pi++) {
        const std::string & prompt = prompts[pi];

        LOG_INF("\n=== Prompt %zu/%zu: \"%s\" ===\n", pi + 1, prompts.size(), prompt.c_str());

        std::vector<llama_token> inp;
        inp = common_tokenize(ctx_tgt, prompt, true, true);

        if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) {
            LOG_INF("note: prompt (%d) exceeds batch (%d) — using chunked prefill\n",
                    (int) inp.size(), llama_n_batch(ctx_tgt));
        }

        struct StepRecord {
            int step;
            int n_draft;
            int n_accept;
            double t0_us;
            double decode_us;
        };
        std::vector<StepRecord> steps;

        int n_predict = 0;
        int n_drafted_total = 0;
        int n_accept_total  = 0;
        bool has_eos = false;

        llama_seq_id seq_id = 0;

        common_sampler_ptr smpl(common_sampler_init(model_tgt, params.sampling));

        // chunked prefill (supports long prompts > batch size)
        {
            const int n_ctx_tokens = (int)inp.size() - 1;
            for (int i = 0; i < n_ctx_tokens; i += 128) {
                const int n_take = std::min(128, n_ctx_tokens - i);
                llama_batch b = llama_batch_init(n_take, 0, 1);
                for (int j = 0; j < n_take; j++) {
                    b.token[j]     = inp.data()[i + j];
                    b.pos[j]       = i + j;
                    b.n_seq_id[j]  = 1;
                    b.seq_id[j][0] = seq_id;
                    b.n_tokens     = n_take;
                }
                llama_decode(ctx_tgt, b);
                llama_decode(ctx_dft.get(), b);
                llama_batch_free(b);
            }
        }

        llama_token id_last = inp.back();

        llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
        prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

        int n_past = inp.size() - 1;

        struct common_speculative * spec = common_speculative_init(params.speculative, 1);
        common_speculative_begin(spec, seq_id, prompt_tgt);

        llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

        size_t n_draft = 0;
        llama_tokens draft;

        const auto t_start = ggml_time_us();

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
            }

            StepRecord rec;

            // Sample and get accepted tokens
            auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);

            if (ids.empty()) break;

            int n_accepted = (int)ids.size() - 1;  // minus the verified token

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

        LOG_INF("--- Prompt %zu stats ---\n", pi + 1);
        LOG_INF("n_predict   = %d\n", n_predict);
        LOG_INF("n_drafted   = %d\n", n_drafted_total);
        LOG_INF("n_accept    = %d\n", n_accept_total);
        LOG_INF("accept_rate = %.1f%%\n",
                n_drafted_total > 0 ? 100.0 * n_accept_total / n_drafted_total : 0.0);
        LOG_INF("total_time  = %.3f s\n", (t_end - t_start) / 1e6);
        LOG_INF("speed       = %.1f t/s\n", n_predict / ((t_end - t_start) / 1e6));

        LOG_INF("\n--- Per-step log (CSV) ---\n");
        LOG_INF("step,n_draft,n_accept,accept_rate,time_us\n");
        for (const auto & s : steps) {
            double ar = s.n_draft > 0 ? 100.0 * s.n_accept / s.n_draft : 0.0;
            LOG_INF("%d,%d,%d,%.1f%%,%.0f\n", s.step, s.n_draft, s.n_accept, ar, s.decode_us);
        }

        common_speculative_free(spec);
        llama_batch_free(batch_tgt);
    }

    llama_backend_free();

    LOG("\n");
    return 0;
}
