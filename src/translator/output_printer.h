#pragma once

#include <vector>

#include "common/options.h"
#include "common/utils.h"
#include "data/alignment.h"
#include "data/vocab.h"
#include "translator/history.h"
#include "translator/hypothesis.h"

namespace marian {

class OutputPrinter {
public:
  std::vector<std::string> getAlts(Result result){
    auto hyp = std::get<1>(result);
    std::string past = "";
    std::vector<std::string> outputs;
    while (hyp) {
      std::stringstream buffer;
      // LOG(info, "{} {}", hyp->alt_words.size());
      if (past.size())
        buffer << past <<" ||| ";
      past = (*vocab_)[hyp->getWord()];
      for (int i = 0;i < hyp->alt_words.size(); i++)
        buffer << " " << (*vocab_)[hyp->alt_words[i]] << " "<<hyp->alt_scores[i];
      outputs.push_back(buffer.str());
      hyp = hyp->getPrevHyp().get();
    }

    std::reverse(outputs.begin(), outputs.end());
    return outputs;
}

  OutputPrinter(Ptr<const Options> options, Ptr<const Vocab> vocab)
      : vocab_(vocab),
        reverse_(options->get<bool>("right-left")),
        nbest_(options->get<bool>("n-best", false)
                   ? options->get<size_t>("beam-size")
                   : 0),
        alignment_(options->get<std::string>("alignment", "")),
        alignmentThreshold_(getAlignmentThreshold(alignment_)),
        wordScores_(options->get<bool>("word-scores")) {}

  template <class OStream>
  void print(Ptr<const History> history, OStream& best1, OStream& bestn) {
    const auto& nbl = history->nBest(nbest_);

    // prepare n-best list output
    for(size_t i = 0; i < nbl.size(); ++i) {
      const auto& result = nbl[i];
      const auto& hypo = std::get<1>(result);
      auto words = std::get<0>(result);

      if(reverse_)
        std::reverse(words.begin(), words.end());

      std::string translation = vocab_->decode(words);
      bestn << history->getLineNum() << " ||| " << translation;

      if(!alignment_.empty())
        bestn << " ||| " << getAlignment(hypo);

      if(wordScores_)
        bestn << " ||| WordScores=" << getWordScores(hypo);

      bestn << " |||";
      if(hypo->getScoreBreakdown().empty()) {
        bestn << " F0=" << hypo->getPathScore();
      } else {
        for(size_t j = 0; j < hypo->getScoreBreakdown().size(); ++j) {
          bestn << " F" << j << "= " << hypo->getScoreBreakdown()[j];
        }
      }

      float realScore = std::get<2>(result);
      bestn << " ||| " << realScore;

      auto outputs = getAlts(result);
      for (auto& s: outputs)
        bestn << std::endl << s;

      if(i < nbl.size() - 1)
        bestn << std::endl;
      else
        bestn << std::flush;
    }

    auto result = history->top();
    auto words = std::get<0>(result);
    
    auto outputs = getAlts(result);
    for (auto& s: outputs)
      best1 << std::endl << s;

    if(reverse_)
      std::reverse(words.begin(), words.end());

    std::string translation = vocab_->decode(words);

//    best1 << translation;
    if(!alignment_.empty()) {
      const auto& hypo = std::get<1>(result);
      best1 << " ||| " << getAlignment(hypo);
    }

    if(wordScores_) {
      const auto& hypo = std::get<1>(result);
      best1 << " ||| WordScores=" << getWordScores(hypo);
    }

    best1 << std::flush;
  }

private:
  Ptr<Vocab const> vocab_;
  bool reverse_{false};            // If it is a right-to-left model that needs reversed word order
  size_t nbest_{0};                // Size of the n-best list to print
  std::string alignment_;          // A non-empty string indicates the type of word alignment
  float alignmentThreshold_{0.f};  // Threshold for converting attention into hard word alignment
  bool wordScores_{false};         // Whether to print word-level scores or not

  // Get word alignment pairs or soft alignment
  std::string getAlignment(const Hypothesis::PtrType& hyp);
  // Get word-level scores
  std::string getWordScores(const Hypothesis::PtrType& hyp);

  float getAlignmentThreshold(const std::string& str) {
    try {
      return std::max(std::stof(str), 0.f);
    } catch(...) {
      return 0.f;
    }
  }
};
}  // namespace marian
