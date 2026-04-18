#include "orchard_solver/cross_section/CrossSection.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace orchard {

namespace {

constexpr double kPi = 3.14159265358979323846;

struct LoopProperties {
    double area {0.0};
    Vec2 centroid {};
    double ix_origin {0.0};
    double iy_origin {0.0};
};

double signedArea(const std::vector<Vec2>& points) {
    double twice_area = 0.0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const auto& a = points[i];
        const auto& b = points[(i + 1) % points.size()];
        twice_area += (a.x * b.y) - (b.x * a.y);
    }
    return 0.5 * twice_area;
}

std::vector<Vec2> ensurePositiveOrientation(std::vector<Vec2> points) {
    if (points.size() < 3U) {
        throw std::runtime_error("A polygon loop needs at least three points");
    }

    if (signedArea(points) < 0.0) {
        std::reverse(points.begin(), points.end());
    }

    return points;
}

std::vector<Vec2> sampleEllipse(const Vec2 center, const Vec2 radii, const int requested_samples) {
    const int samples = std::max(requested_samples, 16);
    std::vector<Vec2> points;
    points.reserve(static_cast<std::size_t>(samples));

    for (int i = 0; i < samples; ++i) {
        const double theta = (2.0 * kPi * static_cast<double>(i)) / static_cast<double>(samples);
        points.push_back(Vec2 {
            center.x + (radii.x * std::cos(theta)),
            center.y + (radii.y * std::sin(theta))
        });
    }

    return points;
}

LoopProperties integrateLoop(std::vector<Vec2> points) {
    points = ensurePositiveOrientation(std::move(points));

    double cross_sum = 0.0;
    double cx_sum = 0.0;
    double cy_sum = 0.0;
    double ix_sum = 0.0;
    double iy_sum = 0.0;

    for (std::size_t i = 0; i < points.size(); ++i) {
        const auto& a = points[i];
        const auto& b = points[(i + 1) % points.size()];
        const double cross = (a.x * b.y) - (b.x * a.y);
        cross_sum += cross;
        cx_sum += (a.x + b.x) * cross;
        cy_sum += (a.y + b.y) * cross;
        ix_sum += ((a.y * a.y) + (a.y * b.y) + (b.y * b.y)) * cross;
        iy_sum += ((a.x * a.x) + (a.x * b.x) + (b.x * b.x)) * cross;
    }

    const double area = 0.5 * cross_sum;
    if (std::abs(area) < 1.0e-12) {
        throw std::runtime_error("Degenerate polygon loop has near-zero area");
    }

    return LoopProperties {
        area,
        Vec2 {cx_sum / (6.0 * area), cy_sum / (6.0 * area)},
        ix_sum / 12.0,
        iy_sum / 12.0
    };
}

LoopProperties regionLoopProperties(const RegionGeometry& geometry) {
    switch (geometry.kind) {
    case SectionShapeKind::SolidEllipse:
        return integrateLoop(sampleEllipse(geometry.center, geometry.radii, geometry.samples));
    case SectionShapeKind::EllipticRing:
    case SectionShapeKind::Polygon:
        break;
    }

    throw std::runtime_error("Use composite ring integration for ring or polygon regions");
}

SectionRegionProperties buildRegionProperties(const TissueRegionDefinition& region, double area, const Vec2 centroid, const double ix_origin, const double iy_origin) {
    const double ix_centroid = std::max(ix_origin - (area * centroid.y * centroid.y), 0.0);
    const double iy_centroid = std::max(iy_origin - (area * centroid.x * centroid.x), 0.0);

    return SectionRegionProperties {
        region.tissue,
        region.material_id,
        area,
        centroid,
        ix_centroid,
        iy_centroid
    };
}

SectionRegionProperties integrateRegion(const TissueRegionDefinition& region) {
    if (region.geometry.kind == SectionShapeKind::SolidEllipse) {
        const auto outer = regionLoopProperties(region.geometry);
        return buildRegionProperties(region, outer.area, outer.centroid, outer.ix_origin, outer.iy_origin);
    }

    std::vector<Vec2> outer_points;
    std::vector<Vec2> inner_points;

    if (region.geometry.kind == SectionShapeKind::EllipticRing) {
        outer_points = sampleEllipse(region.geometry.outer_center, region.geometry.outer_radii, region.geometry.samples);
        inner_points = sampleEllipse(region.geometry.inner_center, region.geometry.inner_radii, region.geometry.samples);
    } else {
        outer_points = region.geometry.outer_points;
        inner_points = region.geometry.inner_points;
    }

    const auto outer = integrateLoop(std::move(outer_points));
    if (inner_points.empty()) {
        return buildRegionProperties(region, outer.area, outer.centroid, outer.ix_origin, outer.iy_origin);
    }

    const auto inner = integrateLoop(std::move(inner_points));
    const double area = outer.area - inner.area;
    if (area <= 0.0) {
        throw std::runtime_error("Ring/polygon region has non-positive area");
    }

    const Vec2 centroid {
        ((outer.area * outer.centroid.x) - (inner.area * inner.centroid.x)) / area,
        ((outer.area * outer.centroid.y) - (inner.area * inner.centroid.y)) / area
    };

    return buildRegionProperties(
        region,
        area,
        centroid,
        outer.ix_origin - inner.ix_origin,
        outer.iy_origin - inner.iy_origin
    );
}

} // namespace

CrossSectionProfile::CrossSectionProfile(const double station)
    : station_(station) {
}

double CrossSectionProfile::station() const noexcept {
    return station_;
}

ParameterizedSectionProfile::ParameterizedSectionProfile(double station, std::vector<TissueRegionDefinition> regions)
    : CrossSectionProfile(station), regions_(std::move(regions)) {
}

SectionProperties ParameterizedSectionProfile::evaluate() const {
    return SectionIntegrator::integrate(regions_);
}

std::string ParameterizedSectionProfile::descriptor() const {
    return "parameterized";
}

ContourSectionProfile::ContourSectionProfile(double station, std::vector<TissueRegionDefinition> regions)
    : CrossSectionProfile(station), regions_(std::move(regions)) {
}

SectionProperties ContourSectionProfile::evaluate() const {
    return SectionIntegrator::integrate(regions_);
}

std::string ContourSectionProfile::descriptor() const {
    return "contour";
}

void MeasuredSectionSeries::addProfile(std::shared_ptr<CrossSectionProfile> profile) {
    profiles_.push_back(std::move(profile));
    std::sort(
        profiles_.begin(),
        profiles_.end(),
        [](const auto& left, const auto& right) { return left->station() < right->station(); }
    );
}

const std::vector<std::shared_ptr<CrossSectionProfile>>& MeasuredSectionSeries::profiles() const noexcept {
    return profiles_;
}

SectionProperties SectionIntegrator::integrate(const std::vector<TissueRegionDefinition>& regions) {
    if (regions.empty()) {
        throw std::runtime_error("Cross-section profile requires at least one tissue region");
    }

    SectionProperties total {};
    double x_weighted = 0.0;
    double y_weighted = 0.0;
    double ix_origin_sum = 0.0;
    double iy_origin_sum = 0.0;

    for (const auto& region : regions) {
        const auto properties = integrateRegion(region);
        total.regions.push_back(properties);
        total.total_area += properties.area;
        x_weighted += properties.area * properties.centroid.x;
        y_weighted += properties.area * properties.centroid.y;
        ix_origin_sum += properties.ix_centroid + (properties.area * properties.centroid.y * properties.centroid.y);
        iy_origin_sum += properties.iy_centroid + (properties.area * properties.centroid.x * properties.centroid.x);
    }

    if (total.total_area <= 0.0) {
        throw std::runtime_error("Cross-section total area must be positive");
    }

    total.centroid = Vec2 {
        x_weighted / total.total_area,
        y_weighted / total.total_area
    };
    total.ix_centroid = std::max(ix_origin_sum - (total.total_area * total.centroid.y * total.centroid.y), 0.0);
    total.iy_centroid = std::max(iy_origin_sum - (total.total_area * total.centroid.x * total.centroid.x), 0.0);

    return total;
}

} // namespace orchard
