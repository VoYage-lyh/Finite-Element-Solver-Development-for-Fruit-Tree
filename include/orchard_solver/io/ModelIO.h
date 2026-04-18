#pragma once

#include <string>

#include "orchard_solver/OrchardModel.h"
#include "orchard_solver/io/SimpleJson.h"

namespace orchard {

class TreeModelBuilder {
public:
    [[nodiscard]] OrchardModel build(const JsonValue& root) const;
};

class ModelValidator {
public:
    void validate(const OrchardModel& model) const;
};

[[nodiscard]] OrchardModel loadModelFromFile(const std::string& file_path);

} // namespace orchard
