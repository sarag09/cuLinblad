#include "culindblad/model_io.hpp"

#include <cmath>
#include <cstddef>
#include <fstream>
#include <map>
#include <numeric>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "culindblad/pulse_envelope.hpp"

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

EnvelopeSpec parse_envelope_spec(const json& e) {
    EnvelopeSpec s;
    s.shape = e.at("shape").get<std::string>();
    s.a = e.value("a", 0.0);
    s.b = e.value("b", 0.0);
    s.T = e.value("T", 0.0);
    s.tr = e.value("tr", 0.0);
    s.t0 = e.value("t0", 0.0);
    s.tau = e.value("tau", 0.0);
    s.sigma = e.value("sigma", 0.0);
    s.omega1 = e.value("omega1", 0.0);
    s.K = e.value("K", 0.0);
    s.cg = e.value("cg", 0.0);
    s.omega_c_idle = e.value("omega_c_idle", 0.0);
    if (e.contains("ugrid")) s.ugrid = e.at("ugrid").get<std::vector<double>>();
    if (e.contains("thetagrid")) s.thetagrid = e.at("thetagrid").get<std::vector<double>>();
    return s;
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

    // Envelopes -> named TimeScalarFunctions (closed-form replay of the Julia gate_sim.jl structs).
    std::map<std::string, TimeScalarFunction> envs;
    if (j.contains("envelopes")) {
        for (auto it = j.at("envelopes").begin(); it != j.at("envelopes").end(); ++it) {
            envs[it.key()] = make_envelope(parse_envelope_spec(it.value()));
        }
    }
    auto lookup_env = [&envs](const std::string& key) -> const TimeScalarFunction& {
        auto e = envs.find(key);
        if (e == envs.end()) {
            throw std::runtime_error("model_io: time-dependent term references unknown envelope '" + key + "'");
        }
        return e->second;
    };

    // Time-dependent Hamiltonian terms: H(t) = H0 + sum_i coeff_i(t) * H_i.
    if (!j.contains("time_dependent_terms")) {
        throw std::runtime_error("model_io: missing 'time_dependent_terms' (may be [])");
    }
    for (const json& t : j.at("time_dependent_terms")) {
        TimeDependentTerm td;
        td.name = t.value("name", std::string());
        td.sites = t.at("sites").get<std::vector<Index>>();
        td.rows = t.at("row_dim").get<Index>();
        td.cols = t.at("col_dim").get<Index>();
        const std::string where = "td term '" + td.name + "'";
        const std::size_t expect = block_dim_over_sites(lm.model.local_dims, td.sites, where);
        if (static_cast<std::size_t>(td.rows) != expect ||
            static_cast<std::size_t>(td.cols) != expect) {
            throw std::runtime_error("model_io: " + where + " row/col_dim != product of site dims "
                                     + std::to_string(expect));
        }
        td.matrix = parse_matrix(t.at("matrix"),
                                 static_cast<std::size_t>(td.rows) * static_cast<std::size_t>(td.cols),
                                 where);
        const json& coeff = t.at("coefficient");
        const std::string kind = coeff.at("kind").get<std::string>();
        if (kind == "envelope") {
            td.coefficient = lookup_env(coeff.at("key").get<std::string>());
        } else if (kind == "product_sqrt") {
            std::vector<TimeScalarFunction> fns;
            for (const json& k : coeff.at("keys")) {
                fns.push_back(lookup_env(k.get<std::string>()));
            }
            td.coefficient = [fns](double t) {
                double r = 1.0;
                for (const TimeScalarFunction& f : fns) r *= std::sqrt(f(t));
                return r;
            };
        } else {
            throw std::runtime_error("model_io: " + where + " unknown coefficient kind '" + kind + "'");
        }
        lm.model.time_dependent_hamiltonian_terms.push_back(std::move(td));
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
