#pragma once

#include <memory>
#include <string>
#include <vector>

#include "orchard_solver/Common.h"

namespace orchard {

enum class SectionShapeKind {
    SolidEllipse,
    EllipticRing,
    Polygon
};

struct RegionGeometry {
    SectionShapeKind kind {SectionShapeKind::SolidEllipse};
    Vec2 center {};
    Vec2 radii {};
    Vec2 outer_center {};
    Vec2 outer_radii {};
    Vec2 inner_center {};
    Vec2 inner_radii {};
    std::vector<Vec2> outer_points;
    std::vector<Vec2> inner_points;
    int samples {48};
};

struct TissueRegionDefinition {
    TissueType tissue {TissueType::Xylem};
    std::string material_id;
    RegionGeometry geometry;
};

struct SectionRegionProperties {
    TissueType tissue {TissueType::Xylem};
    std::string material_id;
    double area {0.0};
    Vec2 centroid {};
    double ix_centroid {0.0};
    double iy_centroid {0.0};
};

struct SectionProperties {
    double total_area {0.0};
    Vec2 centroid {};
    double ix_centroid {0.0};
    double iy_centroid {0.0};
    std::vector<SectionRegionProperties> regions;
};

class CrossSectionProfile {
public:
    explicit CrossSectionProfile(double station);
    virtual ~CrossSectionProfile() = default;

    [[nodiscard]] double station() const noexcept;
    [[nodiscard]] virtual SectionProperties evaluate() const = 0;
    [[nodiscard]] virtual std::string descriptor() const = 0;

private:
    double station_ {0.0};
};

class ParameterizedSectionProfile final : public CrossSectionProfile {
public:
    ParameterizedSectionProfile(double station, std::vector<TissueRegionDefinition> regions);

    [[nodiscard]] SectionProperties evaluate() const override;
    [[nodiscard]] std::string descriptor() const override;

private:
    std::vector<TissueRegionDefinition> regions_;
};

class ContourSectionProfile final : public CrossSectionProfile {
public:
    ContourSectionProfile(double station, std::vector<TissueRegionDefinition> regions);

    [[nodiscard]] SectionProperties evaluate() const override;
    [[nodiscard]] std::string descriptor() const override;

private:
    std::vector<TissueRegionDefinition> regions_;
};

class MeasuredSectionSeries {
public:
    void addProfile(std::shared_ptr<CrossSectionProfile> profile);
    [[nodiscard]] const std::vector<std::shared_ptr<CrossSectionProfile>>& profiles() const noexcept;

private:
    std::vector<std::shared_ptr<CrossSectionProfile>> profiles_;
};

class SectionIntegrator {
public:
    [[nodiscard]] static SectionProperties integrate(const std::vector<TissueRegionDefinition>& regions);
};

} // namespace orchard
