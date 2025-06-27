#include <fmt/format.h>

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numbers>
#include <span>
#include <vector>

constexpr size_t AlignTo256(size_t size) { return (size + 255) & ~size_t(0xff); }

template <typename T>
constexpr size_t AlignedSize() {
    return AlignTo256(sizeof(T));
}

constexpr auto VOCAB_SIZE = 50257;
constexpr auto D_MODEL = 768;
constexpr auto D_SEQ = 1024;
constexpr auto DK = D_MODEL / 12;
constexpr auto EPS = 1e-5f;

// constexpr auto VocabLength = 321428;
constexpr auto VOCAB_LENGTH = 320827;
constexpr auto TEXT_TOKEN_LENGTH = 338024;

struct Token {
    uint32_t offset;
    uint32_t size;
};

struct Data {
    std::array<uint16_t, TEXT_TOKEN_LENGTH> text_token_ids;
};

struct LayerParams {
    float* weight{};
    float* bias{};
};

struct AttentionParams {
    LayerParams c_attn;
    LayerParams c_proj;
};

struct MlpParams {
    LayerParams c_fc;
    LayerParams c_proj;
};

struct TransformerBlock {
    LayerParams ln_1;
    AttentionParams attn;
    LayerParams ln_2;
    MlpParams mlp;
};

struct Parameters {
    LayerParams wte;
    LayerParams wpe;
    std::array<TransformerBlock, 12> headers;
    LayerParams ln_f;
};

struct Activation {
    // Token embedding + position embedding
    float embedding_out[D_SEQ][D_MODEL];

    struct TransformerBlock {
        // LayerNorm 1
        float ln_1_mean[D_SEQ];
        float ln_1_r_std[D_SEQ];
        float ln_1_out[D_SEQ][D_MODEL];
        // Self-Attention: Q, K, V
        //  每个头的维度 | d_k = 64 (隐式) | head_dim = 64 (需要定义) | 64 | DModel
        //  num_heads，即 768 / 12。每个头的 Q, K, V 向量都是64维。
        float attn_c_attn_out[D_SEQ][D_MODEL * 3];

        // float attn_q[DSeq][DModel];
        // float attn_k[DSeq][DModel];
        // float attn_v[DSeq][DModel];

        // logits: Q @ K: [DSeq, DSeq]
        float attn_logits_out[12][D_SEQ][D_SEQ];

        float attn_softmax_out[12][D_SEQ][D_SEQ];
        // Attention score: [DSeq, DSeq]
        float attn_z_out[D_SEQ][D_MODEL];

        // Attention output: [DSeq, DModel]
        float attn_c_proj_out[D_SEQ][D_MODEL];

        // Shortcut connection: [DSeq, DModel] = embedding_out + attn_c_proj_out
        float res_1_out[D_SEQ][D_MODEL];

        // LayerNorm 2
        float ln_2_mean[D_SEQ];
        float ln_2_r_std[D_SEQ];
        float ln_2_out[D_SEQ][D_MODEL];

        // mlp_c_fc_out
        float mlp_c_fc_out[D_SEQ][D_MODEL * 4];
        // gelu
        float mlp_gelu_out[D_SEQ][D_MODEL * 4];
        // mlp_c_proj_out
        float mlp_c_proj_out[D_SEQ][D_MODEL];
        // Shortcut connection: [DSeq, DModel] = attn_c_proj_out + mlp_c_proj_out
        float res_2_out[D_SEQ][D_MODEL];
    };

    std::array<TransformerBlock, 12> blocks;

    float ln_f_mean[D_SEQ];
    float ln_f_r_std[D_SEQ];
    float ln_f_out[D_SEQ][D_MODEL];
    float unembedding_out[D_SEQ][VOCAB_SIZE];
    // Softmax output
    float p[D_SEQ][VOCAB_SIZE];
    // Final layer norm
};

struct ActivationBack {
    struct TransformerBlock {
        // Self-Attention: Q, K, V
        //  每个头的维度 | d_k = 64 (隐式) | head_dim = 64 (需要定义) | 64 | DModel
        //  num_heads，即 768 / 12。每个头的 Q, K, V 向量都是64维。
        float attn_c_attn_out[D_SEQ][D_MODEL * 3];

        // float attn_q[DSeq][DModel];
        // float attn_k[DSeq][DModel];
        // float attn_v[DSeq][DModel];

        // logits: Q @ K: [DSeq, DSeq]
        float attn_logits_out[12][D_SEQ][D_SEQ];

        float attn_softmax_out[12][D_SEQ][D_SEQ];
        // Attention score: [DSeq, DSeq]
        float attn_z_out[D_SEQ][D_MODEL];

        float res_1_out[D_SEQ][D_MODEL];
        float ln_2_out[D_SEQ][D_MODEL];
        float mlp_c_fc_out[D_SEQ][D_MODEL * 4];
        float mlp_gelu_out[D_SEQ][D_MODEL * 4];
        float mlp_c_project_out[D_SEQ][D_MODEL];
        float res_2_out[D_SEQ][D_MODEL];
    };
    std::array<TransformerBlock, 12> blocks;
    float ln_f_out[D_SEQ][D_MODEL];

    float unembedding_out[D_SEQ][VOCAB_SIZE];
};

struct Gradients {
    struct TransformerBlock {
        float ln_1_weight[D_MODEL];
        float ln_1_bias[D_MODEL];

        float attn_c_attn_weight[D_MODEL][D_MODEL * 3];
        float attn_c_attn_bias[D_MODEL * 3];
        float attn_c_proj_weight[D_MODEL][D_MODEL];
        float attn_c_proj_bias[D_MODEL];

        float ln_2_weight[D_MODEL];
        float ln_2_bias[D_MODEL];
        float mlp_c_fc_weight[D_MODEL][D_MODEL * 4];
        float mlp_c_fc_bias[D_MODEL * 4];
        float mlp_c_project_weight[D_MODEL * 4][D_MODEL];
        float mlp_c_project_bias[D_MODEL];
    };
    std::array<TransformerBlock, 12> blocks;

    float ln_f_weight[D_MODEL];
    float ln_f_bias[D_MODEL];

    float wte_weight[VOCAB_SIZE][D_MODEL];
};

class Decoder {
public:
    Decoder() {
        tokens_.resize(VOCAB_SIZE);
        raw_.resize(VOCAB_LENGTH);
        std::ifstream enc{"enc", std::ios::binary};
        enc.read(reinterpret_cast<char*>(tokens_.data()), VOCAB_SIZE * sizeof(Token));
        enc.read(raw_.data(), VOCAB_LENGTH);
    }

    void print() {
        for (const auto& token : tokens_) {
            std::cout << fmt::format(
                "offset:{}, size:{}, token:{}\n",
                token.offset,
                token.size,
                std::string_view{raw_.data() + token.offset, token.size});
        }
    }

    std::string tokens_to_string(std::span<const uint16_t> ids) {
        std::string result;
        for (auto id : ids) {
            // std::cout << fmt::format(
            //     "{}->{}\n", id,
            //     std::string_view{raw.data() + tokens[id].offset, tokens[id].size});
            result += std::string_view{raw_.data() + tokens_[id].offset, tokens_[id].size};
        }
        // std::cout << result << '\n';
        return result;
    }

private:
    // vocab table
    std::vector<Token> tokens_;
    // text
    std::string raw_;
};

// in: [input_size, in_f]
// weight: [in_f, out_f]
// bias: [out_f]
// out: [input_size, out_f]
void full_connect_forward(
    float* in, float* weight, float* bias, float* out, size_t in_f, size_t out_f, size_t input_size) {
    for (size_t q_i = 0; q_i < input_size; q_i++) {
        // j: each output position
        for (size_t j = 0; j < out_f; j++) {
            auto acc = bias ? bias[j] : 0.0f;
            // k_i: each input position
            for (size_t k_i = 0; k_i < in_f; k_i++) {
                acc += in[q_i * in_f + k_i] * weight[k_i * out_f + j];
            }
            out[q_i * out_f + j] = acc;
        }
    }
}

// in: [input_size, in_f]
// weight: [in_f]
// bias: [in_f]
// out: [input_size, in_f]
void layer_norm_forward(
    const float* in,
    const float* weight,
    const float* bias,
    float* mean,
    float* r_std,
    float* out,
    const size_t in_f,
    const size_t input_size) {
    for (size_t i = 0; i < input_size; i++) {
        // Calculate the mean
        // i_th token's vec
        auto x = in + i * in_f;
        float sum = 0.0f;
        for (size_t j = 0; j < in_f; j++) {
            sum += x[j];
        }
        mean[i] = sum / in_f;

        auto total_diff_sq = 0.0f;
        for (size_t j = 0; j < in_f; j++) {
            auto diff = x[j] - mean[i];
            total_diff_sq += diff * diff;
        }
        auto variance = total_diff_sq / in_f;
        r_std[i] = 1.0f / std::sqrt(variance + EPS);
        // r_std[i] = 1.0f / std::sqrt(variance);
        // [DModel]

        auto o = out + i * in_f;
        for (size_t j = 0; j < in_f; j++) {
            auto b = bias ? bias[j] : 0.0f;
            auto ln_in = (x[j] - mean[i]) * r_std[i];
            o[j] = ln_in * weight[j] + b;
        }
    }
}

// in: [input_size, in_f]
// weight: [in_f]
// bias: [in_f]
// out: [input_size, in_f]
void layer_norm_backward(
    const float* in,
    const float* weight,
    const float* g_out,
    const float* mean,
    const float* r_std,
    float* dl_dx,
    float* dl_dweight,
    float* dl_dbias,
    const size_t in_f,
    const size_t input_size) {
    for (size_t i = 0; i < input_size; i++) {
        const auto x_row = in + i * in_f;
        const auto g_out_row = g_out + i * in_f;
        const auto mean_v = mean[i];
        const auto r_std_v = r_std[i];
        const auto gamma_v = weight;
        auto dl_dx_row = dl_dx + i * in_f;

        double sum_g_gamma = 0.0;
        double sum_g_gamma_x_hat = 0.0;
        std::vector<float> x_hat(in_f);
        for (size_t j = 0; j < in_f; j++) {
            x_hat[j] = (x_row[j] - mean_v) * r_std_v;
            dl_dbias[j] += g_out_row[j];
            dl_dweight[j] += g_out_row[j] * x_hat[j];
            auto g_gamma = g_out_row[j] * gamma_v[j];
            sum_g_gamma += g_gamma;
            sum_g_gamma_x_hat += g_gamma * x_hat[j];
        }
        for (size_t j = 0; j < in_f; j++) {
            auto g_gamma = g_out_row[j] * gamma_v[j];
            double term1 = in_f * g_gamma;
            double term2 = sum_g_gamma;
            double term3 = x_hat[j] * sum_g_gamma_x_hat;
            dl_dx_row[j] = static_cast<float>(term1 - term2 - term3) * r_std_v / in_f;
        }
    }
}

void full_connect_backward(
    const float* in,
    const float* weight,
    const float* g_out,
    float* dx,
    float* dl_dweight,
    float* dl_dbias,
    // weight
    const size_t in_f,   //  = 3072
    const size_t out_f,  //  = 768
    const size_t input_size) {
    for (size_t i = 0; i < input_size; ++i) {
        const float* g = g_out + i * out_f;
        float* dx_row = dx + i * in_f;

        for (size_t k = 0; k < in_f; ++k) {  // 先遍历行(in_f)
            float acc = 0.0f;
            for (size_t j = 0; j < out_f; ++j) {      // 再遍历列(out_f)
                acc += g[j] * weight[k * out_f + j];  // W 行优先
            }
            dx_row[k] = acc;
        }
        for (size_t j = 0; j < out_f; ++j) {
            dl_dbias[j] += g[j];
        }
    }
    for (size_t i = 0; i < input_size; ++i) {
        const float* x = in + i * in_f;
        const float* g = g_out + i * out_f;
        for (size_t k = 0; k < in_f; ++k) {
            for (size_t j = 0; j < out_f; ++j) {
                dl_dweight[k * out_f + j] += x[k] * g[j];  // [in_f][out_f]
            }
        }
    }
}

int main() {
    Decoder decode;
    // decode.Print();

    // The text data
    std::ifstream f_data{"data", std::ios::binary};
    std::vector<uint16_t> text_token_ids(TEXT_TOKEN_LENGTH);
    f_data.read(reinterpret_cast<char*>(text_token_ids.data()), TEXT_TOKEN_LENGTH * 2);
    decode.tokens_to_string(text_token_ids);

    std::ifstream f_weight{"model.safetensors", std::ios::binary};
    uint64_t json_size;
    f_weight.read(reinterpret_cast<char*>(&json_size), 8);

    std::string json_str;
    json_str.resize(json_size);
    f_weight.read(reinterpret_cast<char*>(json_str.data()), json_size);
    fmt::print("{}\n", json_str);

    auto safetensors_json = nlohmann::json::parse(json_str);

    // Get tensor data start
    auto tensor_data_start = f_weight.tellg();
    f_weight.seekg(0, std::ios::end);
    auto file_end = f_weight.tellg();
    auto tensor_data_size = file_end - tensor_data_start;
    f_weight.seekg(tensor_data_start);

    std::vector<char> tensor_data(tensor_data_size);
    f_weight.read(tensor_data.data(), tensor_data_size);
    auto tensor_base_ptr = tensor_data.data();

    auto get_tensor = [&](std::string_view key) -> float* {
        if (safetensors_json.contains(key)) {
            auto start = safetensors_json[key]["data_offsets"][0].get<uint64_t>();
            return reinterpret_cast<float*>(tensor_base_ptr + start);
        }
        return nullptr;
    };
    Parameters params;
    params.wte.weight = get_tensor("wte.weight");
    params.wpe.weight = get_tensor("wpe.weight");
    for (int i = 0; i < 12; i++) {
        params.headers[i].ln_1.weight = get_tensor(fmt::format("h.{}.ln_1.weight", i));
        params.headers[i].ln_1.bias = get_tensor(fmt::format("h.{}.ln_1.bias", i));
        params.headers[i].attn.c_attn.weight = get_tensor(fmt::format("h.{}.attn.c_attn.weight", i));
        params.headers[i].attn.c_attn.bias = get_tensor(fmt::format("h.{}.attn.c_attn.bias", i));
        params.headers[i].attn.c_proj.weight = get_tensor(fmt::format("h.{}.attn.c_proj.weight", i));
        params.headers[i].attn.c_proj.bias = get_tensor(fmt::format("h.{}.attn.c_proj.bias", i));
        params.headers[i].ln_2.weight = get_tensor(fmt::format("h.{}.ln_2.weight", i));
        params.headers[i].ln_2.bias = get_tensor(fmt::format("h.{}.ln_2.bias", i));
        params.headers[i].mlp.c_fc.weight = get_tensor(fmt::format("h.{}.mlp.c_fc.weight", i));
        params.headers[i].mlp.c_fc.bias = get_tensor(fmt::format("h.{}.mlp.c_fc.bias", i));
        params.headers[i].mlp.c_proj.weight = get_tensor(fmt::format("h.{}.mlp.c_proj.weight", i));
        params.headers[i].mlp.c_proj.bias = get_tensor(fmt::format("h.{}.mlp.c_proj.bias", i));
    }
    params.ln_f.weight = get_tensor("ln_f.weight");
    params.ln_f.bias = get_tensor("ln_f.bias");

    auto activation = std::make_unique<Activation>();
    size_t input_size = 64;
    // Embedding
    for (size_t i = 0; i < input_size; i++) {
        auto token = text_token_ids[i];
        // Token embedding vec
        auto wte = params.wte.weight + token * D_MODEL;
        // Pos embedding vec
        auto wpe = params.wpe.weight + i * D_MODEL;
        auto embedding_out = activation->embedding_out[i];
        for (size_t j = 0; j < D_MODEL; j++) {
            embedding_out[j] = wte[j] + wpe[j];
        }
    }

    // Transformer blocks

    for (size_t layer_i = 0; layer_i < 12; layer_i++) {
        auto layer_in = layer_i == 0 ? reinterpret_cast<float*>(activation->embedding_out)
                                     : reinterpret_cast<float*>(activation->blocks[layer_i - 1].res_2_out);
        layer_norm_forward(
            layer_in,
            params.headers[layer_i].ln_1.weight,
            params.headers[layer_i].ln_1.bias,
            reinterpret_cast<float*>(activation->blocks[layer_i].ln_1_mean),
            reinterpret_cast<float*>(activation->blocks[layer_i].ln_1_r_std),
            reinterpret_cast<float*>(activation->blocks[layer_i].ln_1_out),
            D_MODEL,
            input_size);

        full_connect_forward(
            reinterpret_cast<float*>(activation->blocks[layer_i].ln_1_out),
            params.headers[layer_i].attn.c_attn.weight,
            params.headers[layer_i].attn.c_attn.bias,
            reinterpret_cast<float*>(activation->blocks[layer_i].attn_c_attn_out),
            D_MODEL,
            D_MODEL * 3,
            input_size);

        for (size_t head_idx = 0; head_idx < 12; head_idx++) {
            for (size_t q_i = 0; q_i < input_size; q_i++) {
                // q_i-th token's q_token_vec:[DK=64]
                auto q = activation->blocks[layer_i].attn_c_attn_out[q_i] + head_idx * DK;
                // auto k = activation->blocks[layer_i].attn_c_attn_out[q_i] + DModel +
                // head_idx * DK; auto v =
                // activation->blocks[layer_i].attn_c_attn_out[q_i] + DModel * 2 +
                // head_idx * DK;

                // Calculate the attention score: q * k_is
                // All the tokens before the q_i-th token
                for (size_t k_i = 0; k_i <= q_i; k_i++) {
                    // k_i-th token's k_vec:[DK=64]
                    auto k = activation->blocks[layer_i].attn_c_attn_out[k_i] + D_MODEL + head_idx * DK;
                    float sum = 0.0f;
                    for (size_t j = 0; j < DK; j++) {
                        sum += q[j] * k[j];
                    }
                    activation->blocks[layer_i].attn_logits_out[head_idx][q_i][k_i] =
                        sum / std::sqrt(static_cast<float>(DK));
                }

                // Softmax
                const double max_logit = *std::max_element(
                    activation->blocks[layer_i].attn_logits_out[head_idx][q_i],
                    activation->blocks[layer_i].attn_logits_out[head_idx][q_i] + q_i + 1);
                const double sum = std::accumulate(
                    activation->blocks[layer_i].attn_logits_out[head_idx][q_i],
                    activation->blocks[layer_i].attn_logits_out[head_idx][q_i] + q_i + 1,
                    0.0,
                    [=](float acc, float logit) { return acc + std::exp(logit - max_logit); });

                const double r_sum = 1.0 / sum;
                for (size_t k_i = 0; k_i <= q_i; k_i++) {
                    auto& x = activation->blocks[layer_i].attn_softmax_out[head_idx][q_i][k_i];
                    auto logit = activation->blocks[layer_i].attn_logits_out[head_idx][q_i][k_i];
                    x = static_cast<float>(std::exp(logit - max_logit) * r_sum);
                }

                for (size_t k_i = q_i + 1; k_i < input_size; k_i++) {
                    activation->blocks[layer_i].attn_softmax_out[head_idx][q_i][k_i] = 0.0f;
                }

                // S @ V: z_token_vec = s_token_vec @ all_v_tokens_vec
                auto s_token = activation->blocks[layer_i].attn_softmax_out[head_idx][q_i];
                for (size_t j = 0; j < DK; j++) {
                    float sum = 0.0f;
                    // attn_softmax_out[head_idx][q_i][q_i, q_i+1, ..., input_size] is zero
                    for (size_t k_i = 0; k_i <= q_i; k_i++) {
                        // k_i-th token's v_token_vec
                        auto v_token = activation->blocks[layer_i].attn_c_attn_out[k_i] + D_MODEL * 2 + head_idx * DK;
                        sum += s_token[k_i] * v_token[j];
                    }
                    activation->blocks[layer_i].attn_z_out[q_i][head_idx * DK + j] = sum;
                }
            }
        }
        // Calculate the attn_c_proj_out
        full_connect_forward(
            reinterpret_cast<float*>(activation->blocks[layer_i].attn_z_out),
            params.headers[layer_i].attn.c_proj.weight,
            params.headers[layer_i].attn.c_proj.bias,
            reinterpret_cast<float*>(activation->blocks[layer_i].attn_c_proj_out),
            D_MODEL,
            D_MODEL,
            input_size);

        {
            // Shortcut connection: [DSeq, DModel] = embedding_out + attn_c_proj_out
            // Here we use the attn_c_proj_out as the shortcut connection output
            auto in_1 = layer_in;
            auto in_2 = reinterpret_cast<float*>(activation->blocks[layer_i].attn_c_proj_out);
            auto out = reinterpret_cast<float*>(activation->blocks[layer_i].res_1_out);
            auto out_end = out + input_size * D_MODEL;
            for (; out != out_end; in_1++, in_2++, out++) {
                *out = *in_1 + *in_2;
            }
        }

        layer_norm_forward(
            reinterpret_cast<float*>(activation->blocks[layer_i].res_1_out),
            params.headers[layer_i].ln_2.weight,
            params.headers[layer_i].ln_2.bias,
            reinterpret_cast<float*>(activation->blocks[layer_i].ln_2_mean),
            reinterpret_cast<float*>(activation->blocks[layer_i].ln_2_r_std),
            reinterpret_cast<float*>(activation->blocks[layer_i].ln_2_out),
            D_MODEL,
            input_size);

        full_connect_forward(
            reinterpret_cast<float*>(activation->blocks[layer_i].ln_2_out),
            params.headers[layer_i].mlp.c_fc.weight,
            params.headers[layer_i].mlp.c_fc.bias,
            reinterpret_cast<float*>(activation->blocks[layer_i].mlp_c_fc_out),
            D_MODEL,
            D_MODEL * 4,
            input_size);

        {
            // mlp_c_fc_out_gelu
            auto in = reinterpret_cast<float*>(activation->blocks[layer_i].mlp_c_fc_out);
            auto out = reinterpret_cast<float*>(activation->blocks[layer_i].mlp_gelu_out);
            auto out_end = out + input_size * D_MODEL * 4;
            for (; out != out_end; in++, out++) {
                auto x = *in;
                // x * 0.5f * (1 + std::erff(x / std::sqrt(2)));
                *out =
                    0.5f * x * (1 + std::tanh(std::sqrt(2 / std::numbers::pi_v<float>) * (x + 0.044715f * x * x * x)));
            }
        }

        full_connect_forward(
            reinterpret_cast<float*>(activation->blocks[layer_i].mlp_gelu_out),
            params.headers[layer_i].mlp.c_proj.weight,
            params.headers[layer_i].mlp.c_proj.bias,
            reinterpret_cast<float*>(activation->blocks[layer_i].mlp_c_proj_out),
            D_MODEL * 4,
            D_MODEL,
            input_size);

        {
            auto in_1 = reinterpret_cast<float*>(activation->blocks[layer_i].res_1_out);
            auto in_2 = reinterpret_cast<float*>(activation->blocks[layer_i].mlp_c_proj_out);
            auto out = reinterpret_cast<float*>(activation->blocks[layer_i].res_2_out);
            auto out_end = out + input_size * D_MODEL;
            for (; out != out_end; in_1++, in_2++, out++) {
                *out = *in_1 + *in_2;
            }
        }
    }

    layer_norm_forward(
        reinterpret_cast<float*>(activation->blocks[11].res_2_out),
        params.ln_f.weight,
        params.ln_f.bias,
        reinterpret_cast<float*>(activation->ln_f_mean),
        reinterpret_cast<float*>(activation->ln_f_r_std),
        reinterpret_cast<float*>(activation->ln_f_out),
        D_MODEL,
        input_size);

    // Here, the wte.weight's shape is [VocabSize, DModel]. Acutally we expect [DModel, VocabSize].

    for (size_t i = 0; i < input_size; i++) {
        auto in = activation->ln_f_out[i];
        auto out = activation->unembedding_out[i];
        // ln_f_out[i] * W.T
        for (size_t j = 0; j < VOCAB_SIZE; j++) {
            float dot = 0.0f;
            const auto weight = params.wte.weight + j * D_MODEL;
            for (size_t k = 0; k < D_MODEL; k++) {
                dot += in[k] * weight[k];
            }
            out[j] = dot;
        }
    }

    {
        // Predict the next token
        // const auto row = activation->unembedding_out[input_size - 1];
        // const uint16_t target_idx = std::distance(row, std::max_element(row, row + VOCAB_SIZE));
        // std::cout << fmt::format("target token id:{}\n", target_idx);
        // for (size_t i = 0; i < input_size; i++) {
        //     uint16_t token_id = text_token_ids[i];
        //     fmt::print("{}", decode.tokens_to_string(std::vector{token_id}));
        // }
        // fmt::print("\n");
        // fmt::print("{}\n", decode.tokens_to_string(std::vector{target_idx}));
    }

    // Softmax
    for (size_t i = 0; i < input_size; i++) {
        auto in = activation->unembedding_out[i];
        auto out = activation->p[i];

        auto max_logit = *std::max_element(in, in + VOCAB_SIZE);
        double exp_sum = 0.0;
        for (size_t j = 0; j < VOCAB_SIZE; ++j) {
            double t = std::exp(static_cast<double>(in[j]) - max_logit);
            out[j] = static_cast<float>(t);
            exp_sum += t;
        }
        auto r_sum = 1.0 / exp_sum;
        for (size_t j = 0; j < VOCAB_SIZE; j++) {
            out[j] = static_cast<float>(out[j] * r_sum);
        }
    }

    // Cross entropy loss
    float loss = 0.0f;
    auto expected_token_ids = std::span{text_token_ids.begin() + 1, text_token_ids.end()};
    for (size_t i = 0; i < input_size; i++) {
        auto true_id = expected_token_ids[i];
        auto ce = -std::logf(activation->p[i][true_id]);
        loss += ce;
    }
    loss /= input_size;
    fmt::print("loss:{}\n", loss);

    // Backward

    // d(loss) / d(softmax(x))
    auto activation_back = std::make_unique<ActivationBack>();
    auto gradients = std::make_unique<Gradients>();
    std::memcpy(activation_back->unembedding_out, activation->p, sizeof(activation->p));
    for (size_t i = 0; i < input_size; i++) {
        auto true_id = expected_token_ids[i];
        activation_back->unembedding_out[i][true_id] -= 1.0f;
        for (size_t j = 0; j < VOCAB_SIZE; j++) {
            activation_back->unembedding_out[i][j] /= input_size;
        }
    }

    {
        float sum = 0.0f;
        for (size_t k = 0; k < input_size; k++) {
            for (size_t j = 0; j < VOCAB_SIZE; j++) {
                sum += activation_back->unembedding_out[k][j];
            }
        }
        std::cout << fmt::format("grad[unembedding_out] sum:{}\n", sum);
    }

    // d(loss) / d(ln_f_out)
    std::memset(activation_back->ln_f_out, 0, sizeof(activation_back->ln_f_out));
    for (size_t i = 0; i < input_size; i++) {
        const auto weight = params.wte.weight;
        // ln_f_out: [D_SEQ, D_MODEL]
        // wte.weight: [VOCAB_SIZE, D_MODEL]
        // unembedding_out: [D_SEQ, VOCAB_SIZE]
        // unembedding_out = ln_f_out @ wte.weight.T
        const auto g_out_row = activation_back->unembedding_out[i];
        // dl_din = dl_out @ wte.weight
        auto dl_din = activation_back->ln_f_out[i];

        for (size_t j = 0; j < D_MODEL; j++) {
            float acc = 0.0f;
            for (size_t k = 0; k < VOCAB_SIZE; k++) {
                acc += g_out_row[k] * weight[k * D_MODEL + j];
            }
            dl_din[j] = acc;
        }
    }
    // d(loss) / d(wte.weight) = d(loss) / d(unembedding_out)  * d(unembedding_out) / d(wte.weight)
    std::memset(gradients->wte_weight, 0, sizeof(gradients->wte_weight));
    for (size_t i = 0; i < input_size; i++) {
        // dl_dweight = dl_out.T @ activation->ln_f_out
        // activation->ln_f_out: [x0, x1, x2, ..., xD_MODEL]
        // dl_dout:[g0, g1, g2, ..., gVOCAB_SIZE]

        // [x0 * g0, x1 * g0, x2 * g0, ..., xD_MODEL * g0]
        // [x0 * g1, x1 * g1, x2 * g1, ..., xD_MODEL * g1]
        // [x0 * g2, x1 * g2, x2 * g2, ..., xD_MODEL * g2]
        // ...
        // [x0 * gVOCAB_SIZE, x1 * gVOCAB_SIZE, x2 * gVOCAB_SIZE, ..., xD_MODEL * gVOCAB_SIZE]
        auto dl_dweight = gradients->wte_weight;                     // [VOCAB_SIZE, D_MODEL]
        const auto x_row = activation->ln_f_out[i];                  // [D_MODEL]
        const auto g_out_row = activation_back->unembedding_out[i];  // [VOCAB_SIZE]
        for (size_t j = 0; j < VOCAB_SIZE; j++) {
            for (size_t k = 0; k < D_MODEL; k++) {
                dl_dweight[j][k] += g_out_row[j] * x_row[k];
            }
        }
    }

    {
        double sum = 0.0;
        for (size_t k = 0; k < input_size; k++) {
            for (size_t j = 0; j < D_MODEL; j++) {
                sum += (double)activation_back->ln_f_out[k][j];
            }
        }
        std::cout << fmt::format("grad[ln_f_out] sum:{}\n", sum);

        sum = 0.0;
        for (size_t k = 0; k < VOCAB_SIZE; k++) {
            for (size_t j = 0; j < D_MODEL; j++) {
                sum += (double)gradients->wte_weight[k][j];
            }
        }
        std::cout << fmt::format("grad[wte.weight] sum:{}\n", sum);
    }

    std::memset(gradients->ln_f_bias, 0, sizeof(gradients->ln_f_bias));
    std::memset(gradients->ln_f_weight, 0, sizeof(gradients->ln_f_weight));
    layer_norm_backward(
        reinterpret_cast<float*>(activation->blocks[11].res_2_out),
        params.ln_f.weight,
        reinterpret_cast<float*>(activation_back->ln_f_out),
        reinterpret_cast<float*>(activation->ln_f_mean),
        reinterpret_cast<float*>(activation->ln_f_r_std),
        reinterpret_cast<float*>(activation_back->blocks[11].res_2_out),
        reinterpret_cast<float*>(gradients->ln_f_weight),
        reinterpret_cast<float*>(gradients->ln_f_bias),
        D_MODEL,
        input_size);

    {
        double sum = 0.0;
        for (size_t i = 0; i < D_MODEL; i++) {
            sum += (double)gradients->ln_f_weight[i];
        }
        std::cout << fmt::format("grad[ln_f_weight] sum:{}\n", sum);
    }
    {
        double sum = 0.0;
        for (size_t i = 0; i < input_size; i++) {
            for (size_t j = 0; j < D_MODEL; j++) {
                sum += (double)activation_back->blocks[11].res_2_out[i][j];
            }
        }
        std::cout << fmt::format("grad[res_2_out] sum:{}\n", sum);
    }

    {
        std::memset(gradients->blocks[11].mlp_c_project_weight, 0, sizeof(gradients->blocks[11].mlp_c_project_weight));
        std::memset(gradients->blocks[11].mlp_c_project_bias, 0, sizeof(gradients->blocks[11].mlp_c_project_bias));
        full_connect_backward(
            reinterpret_cast<float*>(activation->blocks[11].mlp_gelu_out),         // X  [batch, 3072]
            params.headers[11].mlp.c_proj.weight,                                  // W  [3072, 768]
            reinterpret_cast<float*>(activation_back->blocks[11].res_2_out),       // G  [batch, 768]
            reinterpret_cast<float*>(activation_back->blocks[11].mlp_gelu_out),    // dX
            reinterpret_cast<float*>(gradients->blocks[11].mlp_c_project_weight),  // dW
            reinterpret_cast<float*>(gradients->blocks[11].mlp_c_project_bias),    // db
            D_MODEL * 4,                                                           // in_f  = 3072
            D_MODEL,                                                               // out_f = 768
            input_size);
        // for (size_t i = 0; i < input_size; i++) {
        //     const auto weight = params.headers[11].mlp.c_proj.weight;
        //     const auto& g_out_row = activation_back->blocks[11].res_2_out;
        //     auto dl_dx = activation_back->blocks[11].mlp_gelu_out[i];
        //     auto dl_dbias = gradients->blocks[11].mlp_c_project_bias;
        //     for (size_t j = 0; j < D_MODEL * 4; j++) {
        //         float acc = 0.0f;
        //         for (size_t k = 0; k < D_MODEL; k++) {
        //             acc += g_out_row[i][k] * weight[j * D_MODEL + k];
        //         }
        //         dl_dx[j] = acc;
        //     }
        //     for (size_t j = 0; j < D_MODEL; j++) {
        //         dl_dbias[j] += g_out_row[i][j];
        //     }
        // }
        // for (size_t i = 0; i < input_size; i++) {
        //     const auto& g_out_row = activation_back->blocks[11].res_2_out[i];
        //     const auto& x_row = activation->blocks[11].mlp_gelu_out[i];
        //     auto dl_dweight = gradients->blocks[11].mlp_c_project_weight;
        //     for (size_t k = 0; k < D_MODEL * 4; k++) {
        //         for (size_t j = 0; j < D_MODEL; j++) {
        //             dl_dweight[k][j] += x_row[k] * g_out_row[j];
        //         }
        //     }
        // }
    }
    {
        constexpr float kA = 0.7978845608028654f;  // √(2/π)
        constexpr float kCubic = 0.044715f;
        constexpr float kA3Cubic = kA * 3.0f * kCubic;  // a·0.134145

        const size_t in_f = D_MODEL * 4;         // 3072
        const size_t elems = input_size * in_f;  // 64 × 3072

        auto x_ptr = reinterpret_cast<float*>(activation->blocks[11].mlp_c_fc_out);       // x
        auto g_out = reinterpret_cast<float*>(activation_back->blocks[11].mlp_gelu_out);  // ∂L/∂y
        auto g_in = reinterpret_cast<float*>(activation_back->blocks[11].mlp_c_fc_out);   // ∂L/∂x (目标)

        for (size_t idx = 0; idx < elems; ++idx) {
            float x = x_ptr[idx];

            // u = a*(x+0.044715x³)
            float u = kA * (x + kCubic * x * x * x);
            float t = std::tanh(u);      // tanh(u)
            float sech2 = 1.0f - t * t;  // sech²(u)

            // dy/dx
            float dy_dx = 0.5f * (1.0f + t) + 0.5f * x * sech2 * (kA + kA3Cubic * x * x);

            g_in[idx] = g_out[idx] * dy_dx;
        }
    }

    // For test
    {
        double sum = 0.0;
        for (size_t j = 0; j < D_MODEL; j++) {
            sum += std::abs(gradients->blocks[11].mlp_c_project_bias[j]);
        }
        fmt::println("gradients->blocks[11].mlp_c_project_bias sum:{}", sum);
        sum = 0.0;
        for (size_t k = 0; k < D_MODEL * 4; k++) {
            for (size_t j = 0; j < D_MODEL; j++) {
                sum += std::abs(gradients->blocks[11].mlp_c_project_weight[k][j]);
            }
        }
        fmt::println("gradients->blocks[11].mlp_c_project_weight sum: {}", sum);
        sum = 0.0;
        for (size_t k = 0; k < D_SEQ; k++) {
            for (size_t j = 0; j < D_MODEL * 4; j++) {
                sum += std::abs(activation_back->blocks[11].mlp_gelu_out[k][j]);
            }
        }
        fmt::println("activation_back->blocks[11].mlp_gelu_out sum: {}", sum);
        sum = 0.0;
        for (size_t k = 0; k < D_SEQ; k++) {
            for (size_t j = 0; j < D_MODEL * 4; j++) {
                sum += std::abs(activation_back->blocks[11].mlp_c_fc_out[k][j]);
            }
        }
        fmt::println("activation_back->blocks[11].mlp_c_fc_out sum: {}", sum);
    }

    {
        std::memset(gradients->blocks[11].mlp_c_fc_weight, 0, sizeof(gradients->blocks[11].mlp_c_fc_weight));
        std::memset(gradients->blocks[11].mlp_c_fc_bias, 0, sizeof(gradients->blocks[11].mlp_c_fc_bias));
        // in_f = 768 , out_f = 3072
        full_connect_backward(
            reinterpret_cast<float*>(activation->blocks[11].ln_2_out),
            params.headers[11].mlp.c_fc.weight,
            reinterpret_cast<float*>(activation_back->blocks[11].mlp_c_fc_out),
            reinterpret_cast<float*>(activation_back->blocks[11].ln_2_out),
            reinterpret_cast<float*>(gradients->blocks[11].mlp_c_fc_weight),
            reinterpret_cast<float*>(gradients->blocks[11].mlp_c_fc_bias),
            D_MODEL,
            D_MODEL * 4,
            input_size);
    }

    {
        double sum = 0.0;
        for (size_t j = 0; j < D_MODEL * 4; ++j) {  // bias 长度 3072
            sum += std::abs(gradients->blocks[11].mlp_c_fc_bias[j]);
        }
        fmt::println("gradients->blocks[11].mlp_c_fc_bias   sum: {}", sum);

        sum = 0.0;
        for (size_t k = 0; k < D_MODEL; ++k) {
            for (size_t j = 0; j < D_MODEL * 4; ++j) {
                sum += std::abs(gradients->blocks[11].mlp_c_fc_weight[k][j]);
            }
        }
        fmt::println("gradients->blocks[11].mlp_c_fc_weight sum: {}", sum);
    }

    {
        std::memset(gradients->blocks[11].ln_2_weight, 0, sizeof(gradients->blocks[11].ln_2_weight));
        std::memset(gradients->blocks[11].ln_2_bias, 0, sizeof(gradients->blocks[11].ln_2_bias));
        layer_norm_backward(
            reinterpret_cast<float*>(activation->blocks[11].res_1_out),
            params.headers[11].ln_2.weight,
            reinterpret_cast<float*>(activation_back->blocks[11].ln_2_out),
            reinterpret_cast<float*>(activation->blocks[11].ln_2_mean),
            reinterpret_cast<float*>(activation->blocks[11].ln_2_r_std),
            reinterpret_cast<float*>(activation_back->blocks[11].res_1_out),
            reinterpret_cast<float*>(gradients->blocks[11].ln_2_weight),
            reinterpret_cast<float*>(gradients->blocks[11].ln_2_bias),
            D_MODEL,
            input_size);
    }
    {
        double sum = 0.0;
        for (size_t j = 0; j < D_MODEL; ++j) {
            sum += std::abs(gradients->blocks[11].ln_2_bias[j]);
        }
        fmt::println("gradients->blocks[11].ln_2_bias sum: {}", sum);

        sum = 0.0;
        for (size_t k = 0; k < D_MODEL; ++k) {
            sum += std::abs(gradients->blocks[11].ln_2_weight[k]);
        }
        fmt::println("gradients->blocks[11].ln_2_weight sum: {}", sum);
    }

    {
        std::memset(gradients->blocks[11].attn_c_proj_weight, 0, sizeof(gradients->blocks[11].attn_c_proj_weight));
        std::memset(gradients->blocks[11].attn_c_proj_bias, 0, sizeof(gradients->blocks[11].attn_c_proj_bias));
        full_connect_backward(
            reinterpret_cast<float*>(activation->blocks[11].attn_z_out),
            params.headers[11].attn.c_proj.weight,
            reinterpret_cast<float*>(activation_back->blocks[11].res_1_out),
            reinterpret_cast<float*>(activation_back->blocks[11].attn_z_out),
            reinterpret_cast<float*>(gradients->blocks[11].attn_c_proj_weight),
            reinterpret_cast<float*>(gradients->blocks[11].attn_c_proj_bias),
            D_MODEL,
            D_MODEL,
            input_size);
    }
    {
        double sum = 0.0;
        for (size_t j = 0; j < D_MODEL; ++j) {
            sum += std::abs(gradients->blocks[11].attn_c_proj_bias[j]);
        }
        fmt::println("gradients->blocks[11].attn_c_proj_bias sum: {}", sum);

        sum = 0.0;
        for (size_t j = 0; j < D_MODEL; ++j) {
            for (size_t k = 0; k < D_MODEL; ++k) {
                sum += std::abs(gradients->blocks[11].attn_c_proj_weight[j][k]);
            }
        }
        fmt::println("gradients->blocks[11].attn_c_proj_weight sum: {}", sum);

        sum = 0.0;
        for (size_t j = 0; j < D_SEQ; ++j) {
            for (size_t k = 0; k < D_SEQ; ++k) {
                sum += std::abs(activation_back->blocks[11].attn_z_out[j][k]);
            }
        }
        fmt::println("activation_back->blocks[11].attn_z_out sum: {}", sum);

        sum = 0.0;
    }
}
