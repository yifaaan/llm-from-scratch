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

    float ln_f_out[D_SEQ][D_MODEL];
    float y_pred[D_SEQ][VOCAB_LENGTH];
    // Final layer norm
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
void full_connect(float* in, float* weight, float* bias, float* out, size_t in_f, size_t out_f, size_t input_size) {
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
void layer_norm(
    const float* in, const float* weight, const float* bias, float* out, const size_t in_f, const size_t input_size) {
    for (size_t i = 0; i < input_size; i++) {
        // Calculate the mean
        // i_th token's vec
        auto x = in + i * in_f;
        float sum = 0.0f;
        for (size_t j = 0; j < in_f; j++) {
            sum += x[j];
        }
        const auto mean = sum / in_f;

        auto total_diff_sq = 0.0f;
        for (size_t j = 0; j < in_f; j++) {
            auto diff = x[j] - mean;
            total_diff_sq += diff * diff;
        }
        auto variance = total_diff_sq / in_f;
        auto std = std::sqrt(variance + EPS);
        // [DModel]

        auto o = out + i * in_f;
        for (size_t j = 0; j < in_f; j++) {
            auto b = bias ? bias[j] : 0.0f;
            auto ln_in = (x[j] - mean) / std;
            o[j] = ln_in * weight[j] + b;
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
        layer_norm(
            layer_in,
            params.headers[layer_i].ln_1.weight,
            params.headers[layer_i].ln_1.bias,
            reinterpret_cast<float*>(activation->blocks[layer_i].ln_1_out),
            D_MODEL,
            input_size);

        full_connect(
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
                const auto max_logit = *std::max_element(
                    activation->blocks[layer_i].attn_logits_out[head_idx][q_i],
                    activation->blocks[layer_i].attn_logits_out[head_idx][q_i] + q_i + 1);
                const auto sum = std::accumulate(
                    activation->blocks[layer_i].attn_logits_out[head_idx][q_i],
                    activation->blocks[layer_i].attn_logits_out[head_idx][q_i] + q_i + 1,
                    0.0f,
                    [=](float acc, float logit) { return acc + std::exp(logit - max_logit); });

                for (size_t k_i = 0; k_i <= q_i; k_i++) {
                    auto& x = activation->blocks[layer_i].attn_softmax_out[head_idx][q_i][k_i];
                    auto logit = activation->blocks[layer_i].attn_logits_out[head_idx][q_i][k_i];
                    x = std::exp(logit - max_logit) / sum;
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
        full_connect(
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

        layer_norm(
            reinterpret_cast<float*>(activation->blocks[layer_i].res_1_out),
            params.headers[layer_i].ln_2.weight,
            params.headers[layer_i].ln_2.bias,
            reinterpret_cast<float*>(activation->blocks[layer_i].ln_2_out),
            D_MODEL,
            input_size);

        full_connect(
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

        full_connect(
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

    layer_norm(
        reinterpret_cast<float*>(activation->blocks[11].res_2_out),
        params.ln_f.weight,
        params.ln_f.bias,
        reinterpret_cast<float*>(activation->ln_f_out),
        D_MODEL,
        input_size);

    // Here, the wte.weight's shape is [VocabSize, DModel]. Acutally we expect [DModel, VocabSize].

    for (size_t i = 0; i < input_size; i++) {
        auto in = activation->ln_f_out[i];
        auto out = activation->y_pred[i];
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

    // Predict the next token
    const auto row = activation->y_pred[input_size - 1];
    const uint16_t target_idx = std::distance(row, std::max_element(row, row + VOCAB_SIZE));
    std::cout << fmt::format("target token id:{}\n", target_idx);
    for (size_t i = 0; i < input_size; i++) {
        uint16_t token_id = text_token_ids[i];
        fmt::print("{}", decode.tokens_to_string(std::vector{token_id}));
    }
    fmt::print("\n");
    fmt::print("{}\n", decode.tokens_to_string(std::vector{target_idx}));
    // {
    //     float sum = 0.0f;
    //     for (size_t k = 0; k < input_size; k++) {
    //         for (size_t j = 0; j < VOCAB_SIZE; j++) {
    //             sum += activation->y_pred[k][j];
    //         }
    //     }
    //     std::cout << fmt::format("sum:{}\n", sum);
    // }

    // // For test
    // for (size_t i = 0; i < 12; i++) {
    //     float sum = 0.0f;
    //     for (size_t k = 0; k < input_size; k++) {
    //         for (size_t j = 0; j < DModel; j++) {
    //             sum += activation->blocks[i].res_2_out[k][j];
    //         }
    //     }
    //     std::cout << fmt::format("layer:{}, sum:{}\n", i, sum);
    // }
}
