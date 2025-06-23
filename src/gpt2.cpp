#include <array>
#include <cstdint>
#include <fmt/format.h>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <span>
#include <vector>

constexpr size_t AlignTo256(size_t size) {
  return (size + 255) & ~size_t(0xff);
}

template <typename T> constexpr size_t AlignedSize() {
  return AlignTo256(sizeof(T));
}

constexpr auto VocabSize = 50257;
constexpr auto DModel = 768;
constexpr auto DSeq = 1024;
constexpr auto DK = DModel / 12;
constexpr auto EPS = 1e-5f;

// constexpr auto VocabLength = 321428;
constexpr auto VocabLength = 320827;
constexpr auto TextTokenLength = 338024;

struct Token {
  uint32_t offset;
  uint32_t size;
};

struct Data {
  std::array<uint16_t, TextTokenLength> text_token_ids;
};

struct LayerParams {
  float *weight{};
  float *bias{};
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
  float embedding_out[DSeq][DModel];

  struct TransformerBlock {
    // LayerNorm 1
    float ln_1_out[DSeq][DModel];
    // Self-Attention: Q, K, V
    //  每个头的维度 | d_k = 64 (隐式) | head_dim = 64 (需要定义) | 64 | DModel
    //  num_heads，即 768 / 12。每个头的 Q, K, V 向量都是64维。
    float attn_c_attn_out[DSeq][DModel * 3];

    // float attn_q[DSeq][DModel];
    // float attn_k[DSeq][DModel];
    // float attn_v[DSeq][DModel];

    // logits: Q @ K: [DSeq, DSeq]
    float attn_logits_out[12][DSeq][DSeq];

    float attn_softmax_out[12][DSeq][DSeq];
    // Attention score: [DSeq, DSeq]
    float attn_z_out[DSeq][DSeq];

    float attn_ctx_out[DSeq][DModel];
    // Attention output: [DSeq, DModel]
    float attn_c_proj_out[DSeq][DModel];

    // Shortcut connection: [DSeq, DModel] = embedding_out + attn_c_proj_out

    // LayerNorm 2
    float ln_2_out[DSeq][DModel];
  };

  std::array<TransformerBlock, 12> blocks;

  // Final layer norm
};

class Decoder {
public:
  Decoder() {
    tokens.resize(VocabSize);
    raw.resize(VocabLength);
    std::ifstream enc{"enc", std::ios::binary};
    enc.read(reinterpret_cast<char *>(tokens.data()),
             VocabSize * sizeof(Token));
    enc.read(raw.data(), VocabLength);
  }

  void Print() {
    for (const auto &token : tokens) {
      std::cout << fmt::format(
          "offset:{}, size:{}, token:{}\n", token.offset, token.size,
          std::string_view{raw.data() + token.offset, token.size});
    }
  }

  std::string TokensToString(std::span<uint16_t> ids) {
    std::string result;
    for (auto id : ids) {
      // std::cout << fmt::format(
      //     "{}->{}\n", id,
      //     std::string_view{raw.data() + tokens[id].offset, tokens[id].size});
      result +=
          std::string_view{raw.data() + tokens[id].offset, tokens[id].size};
    }
    // std::cout << result << '\n';
    return result;
  }

private:
  // vocab table
  std::vector<Token> tokens;
  // text
  std::string raw;
};

// x: [in_f]
// weight: [in_f, out_f]
// bias: [out_f]
// out: [out_f]
void Linear1XN(const float *x, const float *weight, const float *bias,
               float *out, size_t in_f, size_t out_f) {
  for (size_t j = 0; j < out_f; j++) {
    auto acc = bias ? bias[j] : 0.0f;
    for (size_t i = 0; i < in_f; i++) {
      acc += x[i] * weight[i * out_f + j];
    }
    out[j] = acc;
  }
}

void LayerNorm1XN(float *x, float *weight, float *bias, float *out,
                  size_t in_f) {

  // LayerNorm 2
  // Calculate the mean
  float sum = 0.0;
  for (size_t i = 0; i < in_f; i++) {
    sum += x[i];
  }
  const auto mean = sum / in_f;

  auto total_diff_sq = 0.0;
  for (size_t i = 0; i < in_f; i++) {
    auto diff = x[i] - mean;
    total_diff_sq += diff * diff;
  }
  auto variance = total_diff_sq / in_f;
  auto std = std::sqrt(variance + EPS);
  // [DModel]
  for (size_t i = 0; i < in_f; i++) {
    auto b = bias ? bias[i] : 0.0f;
    auto ln_in = (x[i] - mean) / std;
    out[i] = ln_in * weight[i] + b;
  }
}

int main() {

  Decoder decode;
  // decode.Print();

  // The text data
  std::ifstream f_data{"data", std::ios::binary};
  std::vector<uint16_t> text_token_ids(TextTokenLength);
  f_data.read(reinterpret_cast<char *>(text_token_ids.data()),
              TextTokenLength * 2);
  decode.TokensToString(text_token_ids);

  std::ifstream f_weight{"model.safetensors", std::ios::binary};
  uint64_t json_size;
  f_weight.read(reinterpret_cast<char *>(&json_size), 8);

  std::string json_str;
  json_str.resize(json_size);
  f_weight.read(reinterpret_cast<char *>(json_str.data()), json_size);
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

  auto get_tensor = [&](std::string_view key) -> float * {
    if (safetensors_json.contains(key)) {
      auto start = safetensors_json[key]["data_offsets"][0].get<uint64_t>();
      return reinterpret_cast<float *>(tensor_base_ptr + start);
    }
    return nullptr;
  };
  Parameters params;
  params.wte.weight = get_tensor("wte.weight");
  params.wpe.weight = get_tensor("wpe.weight");
  for (int i = 0; i < 12; i++) {
    params.headers[i].ln_1.weight =
        get_tensor(fmt::format("h.{}.ln_1.weight", i));
    params.headers[i].ln_1.bias = get_tensor(fmt::format("h.{}.ln_1.bias", i));
    params.headers[i].attn.c_attn.weight =
        get_tensor(fmt::format("h.{}.attn.c_attn.weight", i));
    params.headers[i].attn.c_attn.bias =
        get_tensor(fmt::format("h.{}.attn.c_attn.bias", i));
    params.headers[i].attn.c_proj.weight =
        get_tensor(fmt::format("h.{}.attn.c_proj.weight", i));
    params.headers[i].attn.c_proj.bias =
        get_tensor(fmt::format("h.{}.attn.c_proj.bias", i));
    params.headers[i].ln_2.weight =
        get_tensor(fmt::format("h.{}.ln_2.weight", i));
    params.headers[i].ln_2.bias = get_tensor(fmt::format("h.{}.ln_2.bias", i));
    params.headers[i].mlp.c_fc.weight =
        get_tensor(fmt::format("h.{}.mlp.c_fc.weight", i));
    params.headers[i].mlp.c_fc.bias =
        get_tensor(fmt::format("h.{}.mlp.c_fc.bias", i));
    params.headers[i].mlp.c_proj.weight =
        get_tensor(fmt::format("h.{}.mlp.c_proj.weight", i));
    params.headers[i].mlp.c_proj.bias =
        get_tensor(fmt::format("h.{}.mlp.c_proj.bias", i));
  }
  params.ln_f.weight = get_tensor("ln_f.weight");
  params.ln_f.bias = get_tensor("ln_f.bias");

  auto activation = std::make_unique<Activation>();
  size_t input_size = 64;
  // Embedding
  for (size_t i = 0; i < input_size; i++) {
    auto token = text_token_ids[i];
    // Token embedding vec
    auto wte = params.wte.weight + token * DModel;
    // Pos embedding vec
    auto wpe = params.wpe.weight + i * DModel;
    auto embedding_out = activation->embedding_out[i];
    for (size_t j = 0; j < DModel; j++) {
      embedding_out[j] = wte[j] + wpe[j];
    }
  }

  // For test
  // float sum = 0.0;
  // for (size_t i = 0; i < input_size; i++) {
  //   for (size_t j = 0; j < DModel; j++) {
  //     sum += activation->embedding_out[i][j];
  //   }
  // }
  // std::cout << sum << std::endl;

  // Transformer blocks
  size_t layer_idx = 0;
  for (size_t i = 0; i < input_size; i++) {
    // LayerNorm 1
    // Calculate the mean
    // const auto sum =
    //     std::accumulate(std::begin(activation->embedding_out[i]),
    //                     std::end(activation->embedding_out[i]), 0.0f);
    // const auto mean = sum / DModel;

    // auto total_diff_sq = 0.0;
    // for (auto x : activation->embedding_out[i]) {
    //   auto diff = x - mean;
    //   total_diff_sq += diff * diff;
    // }
    // auto variance = total_diff_sq / DModel;
    // auto std = std::sqrt(variance + EPS);
    // // [DModel]
    // auto weight = params.headers[layer_idx].ln_1.weight;
    // auto bias = params.headers[layer_idx].ln_1.bias;
    // for (size_t j = 0; j < DModel; j++) {
    //   auto ln_in = (activation->embedding_out[i][j] - mean) / std;
    //   activation->blocks[layer_idx].ln_1_out[i][j] =
    //       ln_in * weight[j] + bias[j];
    // }
    LayerNorm1XN(activation->embedding_out[i],
                 params.headers[layer_idx].ln_1.weight,
                 params.headers[layer_idx].ln_1.bias,
                 activation->blocks[layer_idx].ln_1_out[i], DModel);
  }

  // For test
  // sum = 0.0;
  // for (size_t i = 0; i < input_size; i++) {
  //   for (size_t j = 0; j < DModel; j++) {
  //     sum += activation->blocks[layer_idx].ln_1_out[i][j];
  //   }
  // }
  // std::cout << sum << std::endl;

  // Every token in the sequence
  for (size_t i = 0; i < input_size; i++) {
    // [DModel, DModel * 3]
    const auto weight = params.headers[layer_idx].attn.c_attn.weight;
    // [DModel * 3]
    const auto bias = params.headers[layer_idx].attn.c_attn.bias;
    // The i_th token's output vector of layerNorm 1:[DModel]
    const auto ln_1_out = activation->blocks[layer_idx].ln_1_out[i];
    // [DModel * 3]
    auto qkv_out = activation->blocks[layer_idx].attn_c_attn_out[i];
    const size_t out_dim = DModel * 3;

    // j: each output position
    Linear1XN(ln_1_out, weight, bias, qkv_out, DModel, out_dim);
    // for (size_t j = 0; j < out_dim; j++) {
    //   auto acc = bias ? bias[j] : 0.0f;
    //   for (size_t k = 0; k < DModel; k++) {
    //     acc += ln_1_out[k] * weight[k * out_dim + j];
    //   }
    //   qkv_out[j] = acc;
    // }

    // Split q, k, v
    // auto& q_out = activation->blocks[layer_idx].attn_q[i];
    // auto& k_out = activation->blocks[layer_idx].attn_k[i];
    // auto& v_out = activation->blocks[layer_idx].attn_v[i];
    // std::copy(qkv_out, qkv_out + DModel, q_out);
    // std::copy(qkv_out + DModel, qkv_out + DModel * 2, k_out);
    // std::copy(qkv_out + DModel * 2, qkv_out + DModel * 3, v_out);
  }

  // For test
  // sum = 0.0;
  // for (size_t i = 0; i < input_size; i++) {
  //   for (size_t j = 0; j < DModel * 3; j++) {
  //     sum += activation->blocks[layer_idx].attn_c_attn_out[i][j];
  //   }
  // }
  // std::cout << sum << std::endl;

  for (size_t head_idx = 0; head_idx < 12; head_idx++) {
    for (size_t q_i = 0; q_i < input_size; q_i++) {
      // q_i-th token's q_token_vec:[DK=64]
      auto q =
          activation->blocks[layer_idx].attn_c_attn_out[q_i] + head_idx * DK;
      // auto k = activation->blocks[layer_idx].attn_c_attn_out[q_i] + DModel +
      // head_idx * DK; auto v =
      // activation->blocks[layer_idx].attn_c_attn_out[q_i] + DModel * 2 +
      // head_idx * DK;

      // Calculate the attention score: q * k_is
      // All the tokens before the q_i-th token
      for (size_t k_i = 0; k_i <= q_i; k_i++) {
        // k_i-th token's k_vec:[DK=64]
        auto k = activation->blocks[layer_idx].attn_c_attn_out[k_i] + DModel +
                 head_idx * DK;
        float sum = 0.0;
        for (size_t j = 0; j < DK; j++) {
          sum += q[j] * k[j];
        }
        activation->blocks[layer_idx].attn_logits_out[head_idx][q_i][k_i] =
            sum / std::sqrt(static_cast<float>(DK));
      }

      // Softmax
      const auto max_logit = *std::max_element(
          activation->blocks[layer_idx].attn_logits_out[head_idx][q_i],
          activation->blocks[layer_idx].attn_logits_out[head_idx][q_i] + q_i +
              1);
      const auto sum = std::accumulate(
          activation->blocks[layer_idx].attn_logits_out[head_idx][q_i],
          activation->blocks[layer_idx].attn_logits_out[head_idx][q_i] + q_i +
              1,
          0.0, [=](float acc, float logit) {
            return acc + std::exp(logit - max_logit);
          });

      for (size_t k_i = 0; k_i <= q_i; k_i++) {
        auto &x =
            activation->blocks[layer_idx].attn_softmax_out[head_idx][q_i][k_i];
        auto logit =
            activation->blocks[layer_idx].attn_logits_out[head_idx][q_i][k_i];
        x = std::exp(logit - max_logit) / sum;
      }

      for (size_t k_i = q_i + 1; k_i < input_size; k_i++) {
        activation->blocks[layer_idx].attn_softmax_out[head_idx][q_i][k_i] =
            0.0;
      }

      // S @ V: z_token_vec = s_token_vec @ all_v_tokens_vec
      auto s_token =
          activation->blocks[layer_idx].attn_softmax_out[head_idx][q_i];
      for (size_t j = 0; j < DK; j++) {
        float sum = 0.0;
        // attn_softmax_out[head_idx][q_i][q_i, q_i+1, ..., input_size] is zero
        for (size_t k_i = 0; k_i <= q_i; k_i++) {
          // k_i-th token's v_token_vec
          auto v_token = activation->blocks[layer_idx].attn_c_attn_out[k_i] +
                         DModel * 2 + head_idx * DK;
          sum += s_token[k_i] * v_token[j];
        }
        activation->blocks[layer_idx].attn_ctx_out[q_i][head_idx * DK + j] =
            sum;
      }
    }

    // Calculate the attn_c_proj_out
    for (size_t q_i = 0; q_i < input_size; q_i++) {
      // [DModel, DModel]
      const auto weight = params.headers[layer_idx].attn.c_proj.weight;
      // [DModel]
      const auto bias = params.headers[layer_idx].attn.c_proj.bias;

      Linear1XN(activation->blocks[layer_idx].attn_ctx_out[q_i], weight, bias,
                activation->blocks[layer_idx].attn_c_proj_out[q_i], DModel,
                DModel);
      // j: each output position
      // for (size_t j = 0; j < DModel; j++) {
      //   auto acc = bias ? bias[j] : 0.0f;
      //   // k_i: each input position
      //   for (size_t k_i = 0; k_i < DModel; k_i++) {
      //     acc += activation->blocks[layer_idx].attn_ctx_out[q_i][k_i] *
      //            weight[k_i * DModel + j];
      //   }
      //   activation->blocks[layer_idx].attn_c_proj_out[q_i][j] = acc;
      // }
    }

    // Shortcut connection: [DSeq, DModel] = embedding_out + attn_c_proj_out
    // Here we use the attn_c_proj_out as the shortcut connection output
    for (size_t i = 0; i < input_size; i++) {
      for (size_t j = 0; j < DModel; j++) {
        activation->blocks[layer_idx].attn_c_proj_out[i][j] +=
            activation->embedding_out[i][j];
      }
    }

    for (size_t i = 0; i < input_size; i++) {
      // LayerNorm 2
      // Calculate the mean
      // const auto sum = std::accumulate(
      //     std::begin(activation->blocks[layer_idx].attn_c_proj_out[i]),
      //     std::end(activation->blocks[layer_idx].attn_c_proj_out[i]), 0.0f);
      // const auto mean = sum / DModel;

      // auto total_diff_sq = 0.0;
      // for (auto x : activation->blocks[layer_idx].attn_c_proj_out[i]) {
      //   auto diff = x - mean;
      //   total_diff_sq += diff * diff;
      // }
      // auto variance = total_diff_sq / DModel;
      // auto std = std::sqrt(variance + EPS);
      // // [DModel]
      // auto weight = params.headers[layer_idx].ln_2.weight;
      // auto bias = params.headers[layer_idx].ln_2.bias;
      // for (size_t j = 0; j < DModel; j++) {
      //   auto ln_in =
      //       (activation->blocks[layer_idx].attn_c_proj_out[i][j] - mean) / std;
      //   activation->blocks[layer_idx].ln_2_out[i][j] =
      //       ln_in * weight[j] + bias[j];
      // }
      LayerNorm1XN(activation->blocks[layer_idx].attn_c_proj_out[i],
                   params.headers[layer_idx].ln_2.weight,
                   params.headers[layer_idx].ln_2.bias,
                   activation->blocks[layer_idx].ln_2_out[i], DModel);
    }
  }
  // For test
  // float sum = 0.0;
  // for (size_t i = 0; i < input_size; i++) {
  //   for (size_t j = 0; j < DModel; j++) {
  //     sum += activation->blocks[layer_idx].attn_ctx_out[i][j];
  //   }
  // }
  // std::cout << sum << std::endl;

  // For test
  float sum = 0.0;
  for (size_t i = 0; i < input_size; i++) {
    for (size_t j = 0; j < DModel; j++) {
      sum += activation->blocks[layer_idx].ln_2_out[i][j];
    }
  }
  std::cout << sum << std::endl;
}