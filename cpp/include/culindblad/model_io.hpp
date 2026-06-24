#pragma once

#include <string>
#include <vector>

#include "culindblad/model.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

// Evolution spec carried alongside the model in the handoff JSON (see JULIA_CPP_HANDOFF_SCHEMA.md).
struct ModelEvolutionSpec {
    double t0 = 0.0;
    double tfinal = 0.0;
    int num_steps = 0;                 // fixed-step count for the custom RK4 (Julia is adaptive; we pin)
    bool rotating_frame = false;
    std::vector<double> frame_omegas;  // bare per-site omega (rad/s), site-0-major; the frame contract
    bool drop_counter_rotating = false;
    std::string result_frame = "lab";  // C++ back-transforms to lab before returning
};

struct ModelInitialStates {
    std::string encoding = "computational";  // currently only "computational"
    int num_states = 0;                      // 2^n_qubits for closed
};

// A deserialized handoff model: the matrix-free Model plus the run metadata the solver needs.
struct LoadedModel {
    Model model;
    bool is_open = false;              // system == "open"  (<=> dissipator_terms non-empty)
    int num_qubits = 0;
    int num_couplers = 0;
    ModelEvolutionSpec evolution;
    ModelInitialStates initial_states;
};

// Parse a term-list JSON file into a LoadedModel. STATIC hamiltonian_terms + dissipator_terms are
// fully populated (row-major Complex blocks from {"real":[...],"imag":[...]}). Time-dependent terms +
// envelopes are wired by the pulse-factory segment; this counts/validates them but does not yet build
// the TimeScalarFunction coefficients. Throws std::runtime_error on malformed input or a failed
// invariant (dims, site ranges, system<->dissipator consistency).
LoadedModel load_model_from_json(const std::string& path);

} // namespace culindblad
