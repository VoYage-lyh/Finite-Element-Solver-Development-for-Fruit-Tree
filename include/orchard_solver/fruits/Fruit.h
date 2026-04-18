#pragma once

#include <string>

namespace orchard {

struct FruitAttachment {
    std::string id;
    std::string branch_id;
    double location_s {1.0};
    double mass {0.0};
    double stiffness {0.0};
    double damping {0.0};
};

} // namespace orchard
