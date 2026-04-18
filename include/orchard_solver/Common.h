#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

namespace orchard {

struct Vec2 {
    double x {0.0};
    double y {0.0};
};

struct Vec3 {
    double x {0.0};
    double y {0.0};
    double z {0.0};
};

inline double distance(const Vec3& a, const Vec3& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    const double dz = a.z - b.z;
    return std::sqrt((dx * dx) + (dy * dy) + (dz * dz));
}

enum class TissueType {
    Xylem,
    Pith,
    Phloem
};

inline std::string toString(const TissueType tissue) {
    switch (tissue) {
    case TissueType::Xylem:
        return "xylem";
    case TissueType::Pith:
        return "pith";
    case TissueType::Phloem:
        return "phloem";
    }

    throw std::runtime_error("Unsupported tissue type");
}

inline TissueType tissueTypeFromString(const std::string& value) {
    if (value == "xylem") {
        return TissueType::Xylem;
    }

    if (value == "pith") {
        return TissueType::Pith;
    }

    if (value == "phloem") {
        return TissueType::Phloem;
    }

    throw std::runtime_error("Unknown tissue type: " + value);
}

} // namespace orchard
