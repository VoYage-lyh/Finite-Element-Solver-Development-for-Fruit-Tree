#pragma once

#include <unordered_map>
#include <vector>

#include "orchard_solver/discretization/Assembler.h"

namespace orchard {

[[nodiscard]] std::unordered_map<std::string, std::vector<double>> computeGravityAxialForces(
    const AssembledModel& assembled,
    const std::vector<double>& gravity_load
);

} // namespace orchard
