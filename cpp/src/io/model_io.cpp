#include "culindblad/model_io.hpp"

#include <cstddef>
#include <fstream>
#include <numeric>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace culindblad {

using json = nlohmann::json;

namespace {

// Decode a {"real":[...],"imag":[...]} block into a row-major Complex vector of the expected length.
std::vector<Complex> parse_matrix(const json& m, std::size_t expected, const std::string& where) {
    const json& re = m.at("real");
    const json& im = m.at("imag");
    if (!re.is_array() || !im.is_array() || re.size() != im.size()) {
        throw std::runtime_error("model_io: " + where + " real/imag must be equal-length arrays");
    }
    if (re.size() != expected) {
        throw std::runtime_error("model_io: " + where + " matrix length " + std::to_string(re.size())
                                 + " != row_dim*col_dim " + std::to_string(expected));
    }
    std::vector<Complex> out(re.size());
    for (std::size_t k = 0; k < re.size(); ++k) {
        out[k] = Complex(re[k].get<double>(), im[k].get<double>());
    }
    return out;
}

// Product of local_dims over a term's sites == the expected (square) block dimension.
std::size_t block_dim_over_sites(const std::vector<Index>& local_dims,
                                 const std::vector<Index>& sites,
                                 const std::string& where) {
    std::size_t d = 1;
    for (const Index s : sites) {
        if (s >= local_dims.size()) {
            throw std::runtime_error("model_io: " + where + " site index " + std::to_string(s)
                                     + " out of range (local_dims size " + std::to_string(local_dims.size()) + ")");
        }
        d *= static_cast<std::size_t>(local_dims[s]);
    }
    return d;
}

OperatorTerm parse_term(const json& t, TermKind kind, const std::vector<Index>& local_dims) {
    OperatorTerm term;
    term.kind = kind;
    term.name = t.value("name", std::string());
    term.sites = t.at("sites").get<std::vector<Index>>();
    term.row_dim = t.at("row_dim").get<Index>();
    term.col_dim = t.at("col_dim").get<Index>();
    const std::string where = "term '" + term.name + "'";
    // Block must be square and equal to the product of the site dims (site-0-major k-site block).
    const std::size_t expect = block_dim_over_sites(local_dims, term.sites, where);
    if (static_cast<std::size_t>(term.row_dim) != expect ||
        static_cast<std::size_t>(term.col_dim) != expect) {
        throw std::runtime_error("model_io: " + where + " row/col_dim ("
                                 + std::to_string(term.row_dim) + "," + std::to_string(term.col_dim)
                                 + ") != product of site dims " + std::to_string(expect));
    }
    term.matrix = parse_matrix(t.at("matrix"),
                               static_cast<std::size_t>(term.row_dim) * static_cast<std::size_t>(term.col_dim),
                               where);
    return term;
}

} // namespace

LoadedModel load_model_from_json(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("model_io: cannot open model file: " + path);
    }
    json j;
    try {
        f >> j;
    } catch (const json::parse_error& e) {
        throw std::runtime_error(std::string("model_io: JSON parse error: ") + e.what());
    }

    LoadedModel lm;

    const json& dev = j.at("device");
    lm.model.local_dims = dev.at("local_dims").get<std::vector<Index>>();
    if (lm.model.local_dims.empty()) {
        throw std::runtime_error("model_io: device.local_dims is empty");
    }
    lm.num_qubits = dev.value("n_qubits", 0);
    lm.num_couplers = dev.value("n_couplers", 0);

    const std::string system = j.at("system").get<std::string>();
    if (system != "closed" && system != "open") {
        throw std::runtime_error("model_io: system must be 'closed' or 'open', got '" + system + "'");
    }
    lm.is_open = (system == "open");

    for (const json& t : j.at("hamiltonian_terms")) {
        lm.model.hamiltonian_terms.push_back(parse_term(t, TermKind::Hamiltonian, lm.model.local_dims));
    }
    for (const json& t : j.at("dissipator_terms")) {
        lm.model.dissipator_terms.push_back(parse_term(t, TermKind::Dissipator, lm.model.local_dims));
    }

    // Invariant: system == "closed"  iff  no dissipators.
    if (lm.is_open == lm.model.dissipator_terms.empty()) {
        throw std::runtime_error("model_io: system='" + system + "' inconsistent with "
                                 + std::to_string(lm.model.dissipator_terms.size()) + " dissipator_terms");
    }

    // Time-dependent terms + envelopes: wired by the pulse-factory segment. Require the keys to exist
    // (so a malformed file is caught) but do not yet build coefficients.
    if (!j.contains("time_dependent_terms")) {
        throw std::runtime_error("model_io: missing 'time_dependent_terms' (may be [])");
    }

    const json& ev = j.at("evolution");
    lm.evolution.t0 = ev.value("t0", 0.0);
    lm.evolution.tfinal = ev.at("tfinal").get<double>();
    lm.evolution.num_steps = ev.value("num_steps", 0);
    lm.evolution.rotating_frame = ev.value("rotating_frame", false);
    if (ev.contains("frame_omegas")) {
        lm.evolution.frame_omegas = ev.at("frame_omegas").get<std::vector<double>>();
        if (lm.evolution.frame_omegas.size() != lm.model.local_dims.size()) {
            throw std::runtime_error("model_io: frame_omegas length != number of sites");
        }
    }
    lm.evolution.drop_counter_rotating = ev.value("drop_counter_rotating", false);
    lm.evolution.result_frame = ev.value("result_frame", std::string("lab"));

    const json& is = j.at("initial_states");
    lm.initial_states.encoding = is.value("encoding", std::string("computational"));
    lm.initial_states.num_states = is.value("num_states", 0);

    return lm;
}

} // namespace culindblad
