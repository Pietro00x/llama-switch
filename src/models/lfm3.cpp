#include "models.h"

#include "../llama-memory-hybrid-iswa.h"

// LFM3: LFM2-style hybrid of gated short convolutions and attention, where the attention layers alternate
// between sliding window and full attention. Compared to LFM2 it adds sandwich norms, a per-head sigmoid
// gate on the attention output, partial RoPE on the full attention layers (a rope_freqs mask) and a
// separate RoPE base for the sliding window layers.
llm_build_lfm3::llm_build_lfm3(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    auto build_attn_block = [&model, this](ggml_tensor *                  cur_in,
                                           ggml_tensor *                  inp_pos,
                                           llm_graph_input_attn_kv_iswa * inp_attn,
                                           int                            il) -> ggml_tensor * {
        const int64_t n_embd_head = hparams.n_embd_head_k;
        const int64_t n_head_kv   = hparams.n_head_kv(il);

        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        // partial RoPE on the full attention layers: the masked frequency pairs are left unrotated
        ggml_tensor * rope_factors = hparams.is_swa(il) ? nullptr : model.layers[il].rope_freqs;

        auto * q = build_lora_mm(model.layers[il].wq, cur_in);
        cb(q, "model.layers.{}.self_attn.q_proj", il);
        auto * k = build_lora_mm(model.layers[il].wk, cur_in);
        cb(k, "model.layers.{}.self_attn.k_proj", il);
        auto * v = build_lora_mm(model.layers[il].wv, cur_in);
        cb(v, "model.layers.{}.self_attn.v_proj", il);

        q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head,    n_tokens);
        k = ggml_reshape_3d(ctx0, k, n_embd_head, n_head_kv, n_tokens);
        v = ggml_reshape_3d(ctx0, v, n_embd_head, n_head_kv, n_tokens);

        q = build_norm(q, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
        cb(q, "model.layers.{}.self_attn.q_layernorm", il);
        k = build_norm(k, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
        cb(k, "model.layers.{}.self_attn.k_layernorm", il);

        q = ggml_rope_ext(ctx0, q, inp_pos, rope_factors, n_rot, rope_type, n_ctx_orig,
                          freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);
        k = ggml_rope_ext(ctx0, k, inp_pos, rope_factors, n_rot, rope_type, n_ctx_orig,
                          freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);

        ggml_tensor * attn_out = build_attn(inp_attn,
                NULL, NULL,
                q, k, v, nullptr, nullptr, nullptr, 1.0f / sqrtf(float(n_embd_head)), il);
        cb(attn_out, "attn_out", il);

        // per-head sigmoid gate, computed from the attention input
        ggml_tensor * gate = build_lora_mm(model.layers[il].wqkv_gate, cur_in);
        gate = ggml_sigmoid(ctx0, gate);
        cb(gate, "model.layers.{}.self_attn.gate", il);

        ggml_tensor * attn_3d = ggml_reshape_3d(ctx0, attn_out, n_embd_head, n_head, n_tokens);
        ggml_tensor * gate_3d = ggml_reshape_3d(ctx0, gate,     1,           n_head, n_tokens);
        attn_3d  = ggml_mul(ctx0, attn_3d, gate_3d);
        attn_out = ggml_reshape_2d(ctx0, attn_3d, n_embd_head * n_head, n_tokens);

        ggml_tensor * out = build_lora_mm(model.layers[il].wo, attn_out);
        cb(out, "model.layers.{}.self_attn.out_proj", il);

        return out;
    };
    auto build_shortconv_block = [&model, this](ggml_tensor *        cur,
                                                llm_graph_input_rs * inp_recr,
                                                int                  il) -> ggml_tensor * {
        const auto * mctx_cur = static_cast<const llama_memory_hybrid_iswa_context *>(mctx)->get_recr();
        const uint32_t kv_head      = mctx_cur->get_head();
        const int64_t  n_seq_tokens = ubatch.n_seq_tokens;
        const int64_t  n_seqs       = ubatch.n_seqs;
        GGML_ASSERT(n_seqs != 0);
        GGML_ASSERT(ubatch.equal_seqs());
        GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

        GGML_ASSERT(hparams.n_shortconv_l_cache > 1);
        const uint32_t d_conv = hparams.n_shortconv_l_cache - 1;

        // {n_embd, n_tokens} => {n_embd, n_seq_tokens, n_seqs}
        cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);

        auto * bcx = build_lora_mm(model.layers[il].shortconv.in_proj, cur);
        cb(bcx, "model.layers.{}.conv.in_proj", il);

        constexpr auto n_chunks = 3;
        GGML_ASSERT(bcx->ne[0] % n_chunks == 0);
        const auto chunk_size = bcx->ne[0] / n_chunks;
        auto *     b          = ggml_view_3d(ctx0, bcx, chunk_size, bcx->ne[1], bcx->ne[2], bcx->nb[1], bcx->nb[2],
                                             0 * chunk_size * ggml_element_size(bcx));
        auto *     c          = ggml_view_3d(ctx0, bcx, chunk_size, bcx->ne[1], bcx->ne[2], bcx->nb[1], bcx->nb[2],
                                             1 * chunk_size * ggml_element_size(bcx));
        auto *     x          = ggml_view_3d(ctx0, bcx, chunk_size, bcx->ne[1], bcx->ne[2], bcx->nb[1], bcx->nb[2],
                                             2 * chunk_size * ggml_element_size(bcx));

        auto * bx = ggml_transpose(ctx0, ggml_mul(ctx0, b, x));

        // read conv state
        auto * conv_state = mctx_cur->get_r_l(il);
        auto * conv_rs    = build_rs(inp_recr, conv_state, hparams.n_embd_r(), n_seqs);
        auto * conv       = ggml_reshape_3d(ctx0, conv_rs, d_conv, hparams.n_embd, n_seqs);

        bx = ggml_concat(ctx0, conv, bx, 0);
        GGML_ASSERT(bx->ne[0] > conv->ne[0]);

        // last d_conv columns is a new conv state
        auto * new_conv = ggml_view_3d(ctx0, bx, conv->ne[0], bx->ne[1], bx->ne[2], bx->nb[1], bx->nb[2],
                                       (bx->ne[0] - conv->ne[0]) * ggml_element_size(bx));
        GGML_ASSERT(ggml_are_same_shape(conv, new_conv));

        // write new conv conv state
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, new_conv,
                                               ggml_view_1d(ctx0, conv_state, ggml_nelements(new_conv),
                                                            kv_head * d_conv * n_embd * ggml_element_size(new_conv))));

        auto * conv_kernel = model.layers[il].shortconv.conv;
        auto * conv_out    = ggml_ssm_conv(ctx0, bx, conv_kernel);
        cb(conv_out, "model.layers.{}.conv.conv", il);

        auto * y = ggml_mul(ctx0, c, conv_out);
        y        = build_lora_mm(model.layers[il].shortconv.out_proj, y);
        cb(y, "model.layers.{}.conv.out_proj", il);
        // {n_embd, n_seq_tokens, n_seqs} => {n_embd, n_tokens}
        y = ggml_reshape_2d(ctx0, y, y->ne[0], n_seq_tokens * n_seqs);

        return y;
    };

    ggml_tensor * cur = build_inp_embd(model.tok_embd);
    cb(cur, "model.embed_tokens", -1);

    ggml_build_forward_expand(gf, cur);

    auto * inp_hybrid = build_inp_mem_hybrid_iswa();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        auto * prev_cur = cur;

        cur = build_norm(cur, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "model.layers.{}.operator_norm", il);

        cur = hparams.is_recurrent(il) ? build_shortconv_block(cur, inp_hybrid->get_recr(), il) :
                                         build_attn_block(cur, inp_pos, inp_hybrid->get_attn(), il);

        if (model.layers[il].attn_post_norm) {
            cur = build_norm(cur, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "model.layers.{}.post_operator_norm", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur      = ggml_get_rows(ctx0, cur, inp_out_ids);
            prev_cur = ggml_get_rows(ctx0, prev_cur, inp_out_ids);
        }

        cur = ggml_add(ctx0, prev_cur, cur);

        auto * ffn_norm_out = build_norm(cur, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(ffn_norm_out, "model.layers.{}.ffn_norm", il);

        ggml_tensor * ffn_out = build_ffn(ffn_norm_out,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_out, "model.layers.{}.ffn_out", il);

        if (model.layers[il].ffn_post_norm) {
            ffn_out = build_norm(ffn_out, model.layers[il].ffn_post_norm, NULL, LLM_NORM_RMS, il);
            cb(ffn_out, "model.layers.{}.post_ffn_norm", il);
        }

        cur = ggml_add(ctx0, cur, ffn_out);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
    }

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);

    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
