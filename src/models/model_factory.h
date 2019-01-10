#pragma once

#include "marian.h"

#include "layers/factory.h"
#include "models/encoder_decoder.h"

namespace marian {
namespace models {

class EncoderFactory : public Factory {
public:
  EncoderFactory(Ptr<ExpressionGraph> graph = nullptr) : Factory() {}

  virtual Ptr<EncoderBase> construct(Ptr<ExpressionGraph> graph);
};

typedef Accumulator<EncoderFactory> encoder;

class DecoderFactory : public Factory {
public:
  DecoderFactory(Ptr<ExpressionGraph> graph = nullptr) : Factory() {}

  virtual Ptr<DecoderBase> construct(Ptr<ExpressionGraph> graph);
};

typedef Accumulator<DecoderFactory> decoder;

class EncoderDecoderFactory : public Factory {
private:
  std::vector<encoder> encoders_;
  std::vector<decoder> decoders_;

public:
  EncoderDecoderFactory(Ptr<ExpressionGraph> graph = nullptr)
      : Factory() {}

  Accumulator<EncoderDecoderFactory> push_back(encoder enc) {
    encoders_.push_back(enc);
    return Accumulator<EncoderDecoderFactory>(*this);
  }

  Accumulator<EncoderDecoderFactory> push_back(decoder dec) {
    decoders_.push_back(dec);
    return Accumulator<EncoderDecoderFactory>(*this);
  }

  virtual Ptr<ModelBase> construct(Ptr<ExpressionGraph> graph);
};

typedef Accumulator<EncoderDecoderFactory> encoder_decoder;

Ptr<ModelBase> by_type(std::string type, usage, Ptr<Options> options);

Ptr<ModelBase> from_options(Ptr<Options> options, usage);

Ptr<ModelBase> from_config(Ptr<Config> config, usage);
}  // namespace models
}  // namespace marian
