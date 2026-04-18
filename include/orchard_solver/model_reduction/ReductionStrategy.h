#pragma once

#include <string>
#include <vector>

namespace orchard {

struct ReducedBasis {
    std::vector<std::vector<double>> vectors;
};

class ReductionStrategy {
public:
    virtual ~ReductionStrategy() = default;
    [[nodiscard]] virtual std::string name() const = 0;
};

} // namespace orchard
