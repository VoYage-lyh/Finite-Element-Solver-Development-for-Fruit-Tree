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

inline Vec3 operator+(const Vec3& left, const Vec3& right) {
    return Vec3 {left.x + right.x, left.y + right.y, left.z + right.z};
}

inline Vec3 operator-(const Vec3& left, const Vec3& right) {
    return Vec3 {left.x - right.x, left.y - right.y, left.z - right.z};
}

inline Vec3 operator*(const Vec3& value, const double scale) {
    return Vec3 {value.x * scale, value.y * scale, value.z * scale};
}

inline Vec3 operator*(const double scale, const Vec3& value) {
    return value * scale;
}

inline Vec3 operator/(const Vec3& value, const double scale) {
    return Vec3 {value.x / scale, value.y / scale, value.z / scale};
}

inline double distance(const Vec3& a, const Vec3& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    const double dz = a.z - b.z;
    return std::sqrt((dx * dx) + (dy * dy) + (dz * dz));
}

inline double dot(const Vec3& left, const Vec3& right) {
    return (left.x * right.x) + (left.y * right.y) + (left.z * right.z);
}

inline Vec3 cross(const Vec3& left, const Vec3& right) {
    return Vec3 {
        (left.y * right.z) - (left.z * right.y),
        (left.z * right.x) - (left.x * right.z),
        (left.x * right.y) - (left.y * right.x)
    };
}

inline double norm(const Vec3& value) {
    return std::sqrt(dot(value, value));
}

inline Vec3 normalize(const Vec3& value) {
    const double value_norm = norm(value);
    if (value_norm <= 1.0e-12) {
        throw std::runtime_error("Cannot normalize a near-zero vector");
    }

    return value / value_norm;
}

inline Vec3 lerp(const Vec3& left, const Vec3& right, const double alpha) {
    return ((1.0 - alpha) * left) + (alpha * right);
}

inline double lerp(const double left, const double right, const double alpha) {
    return ((1.0 - alpha) * left) + (alpha * right);
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
