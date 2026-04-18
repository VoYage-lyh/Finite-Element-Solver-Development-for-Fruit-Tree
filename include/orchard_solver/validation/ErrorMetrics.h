#pragma once

namespace orchard {

struct ErrorMetrics {
    double modal_error {0.0};
    double response_rms_error {0.0};
    double peak_error {0.0};
    double runtime_speedup {1.0};
};

} // namespace orchard
