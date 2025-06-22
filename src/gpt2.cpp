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
  struct Embedding {
    float out[DSeq][DModel];
  };
  
  struct LayerNorm {
    float out[DSeq][DModel];
  };

  Embedding embedding;
  LayerNorm ln_1;
  LayerNorm ln_2;
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

  // get tensor data start
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
  for (size_t i = 0; i < input_size; i++) {
    auto token = text_token_ids[i];
    // Token embedding vec
    auto wte = params.wte.weight + token * DModel;
    // Pos embedding vec
    auto wpe = params.wpe.weight + i * DModel;
    auto embedding_out = activation->embedding.out[i];
    for (size_t j = 0; j < DModel; j++) {
      embedding_out[j] = wte[j] + wpe[j];
    }
  }
  float sum = 0.0;
  for (size_t i = 0; i < input_size; i++) {
    for (size_t j = 0; j < DModel; j++) {
      sum += activation->embedding.out[i][j];
    }
  }
  std::cout << sum << std::endl;


  size_t layer_idx = 0;
  for (size_t i = 0; i < input_size; i++) {
    // Calculate the mean
    const auto sum = std::accumulate(std::begin(activation->embedding.out[i]), std::end(activation->embedding.out[i]), 0.0f);
    const auto mean = sum / DModel;

    auto total_diff_sq = 0.0;
    for (auto x : activation->embedding.out[i]) {
      auto diff = x - mean;
      total_diff_sq += diff * diff;
    }
    auto variance = total_diff_sq / DModel;
    const float eps = 1e-5f;
    auto std = std::sqrt(variance + eps);

    auto weight = params.headers[layer_idx].ln_1.weight;
    auto bias = params.headers[layer_idx].ln_1.bias;
    for (size_t j = 0; j < DModel; j++) {
      auto ln_in = (activation->embedding.out[i][j] - mean) / std;
      activation->ln_1.out[i][j] = ln_in * weight[j] + bias[j];
    }
  }

  sum = 0.0;
  for (size_t i = 0; i < input_size; i++) {
    for (size_t j = 0; j < DModel; j++) {
      sum += activation->ln_1.out[i][j];
    }
  }
  std::cout << sum << std::endl;
}