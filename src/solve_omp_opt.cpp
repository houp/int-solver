#include <omp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

struct Row {
    std::vector<int> vars;
    std::vector<int> coeffs;
    int rhs = 0;
};

struct Model {
    int nvars = 0;
    int nrows = 0;
    std::vector<int> row_offsets;
    std::vector<int> all_vars;
    std::vector<int> all_coeffs;
    std::vector<int> rhs;
    std::vector<std::vector<std::pair<int, int>>> var_adj;
    std::vector<int> var_order;
    std::vector<int> row_max_coeff;
    std::vector<std::pair<int, unsigned char>> fixed_assignments;
    std::vector<std::vector<std::pair<int, unsigned char>>> implications_when_zero;
    std::vector<std::vector<std::pair<int, unsigned char>>> implications_when_one;
};

struct PropertyInput {
    std::string name;
    std::string path;
};

struct Options {
    std::string base_property;
    std::vector<PropertyInput> properties;
    int threads = std::max(1u, std::thread::hardware_concurrency());
    int spawn_depth = 8;
    uint64_t report_every = 1000000;
    bool count_only = false;
    bool write_nonzero_masks_only = false;
    std::string out_file;
    std::string metadata_file;
};

struct RelationCheck {
    int a = 0;
    int b = 0;
    unsigned char xor_value = 0;
};

struct FixedCheck {
    int var = 0;
    unsigned char value = 0;
};

struct ImplicationCheck {
    int lhs = 0;
    int rhs = 0;
};

struct CompiledProperty {
    std::string name;
    std::string path;
    int nrows = 0;
    bool always_false = false;
    std::vector<FixedCheck> fixed_checks;
    std::vector<RelationCheck> relation_checks;
    std::vector<ImplicationCheck> implication_checks;
    std::vector<Row> generic_rows;

    bool check(const std::vector<signed char>& x) const {
        if (always_false) return false;

        for (const auto& check : fixed_checks) {
            if (x[check.var] != static_cast<signed char>(check.value)) return false;
        }
        for (const auto& check : relation_checks) {
            if (((x[check.a] ^ x[check.b]) & 1) != check.xor_value) return false;
        }
        for (const auto& check : implication_checks) {
            if (x[check.lhs] == 1 && x[check.rhs] == 0) return false;
        }
        for (const auto& row : generic_rows) {
            int sum = 0;
            for (std::size_t k = 0; k < row.vars.size(); ++k) {
                sum += row.coeffs[k] * static_cast<int>(x[row.vars[k]]);
            }
            if (sum != row.rhs) return false;
        }
        return true;
    }
};

struct Runtime {
    bool write_output = false;
    bool write_nonzero_masks_only = false;
    uint64_t report_every = 1000000;
    int nvars = 0;
    std::size_t solution_bytes = 0;
    std::size_t mask_bytes = 0;
    std::size_t record_bytes = 0;
    std::string out_file;
    std::string metadata_file;
    std::vector<CompiledProperty> properties;
    std::vector<std::string> shard_files;
};

struct WorkerState {
    const Model* M;
    std::vector<signed char> x;
    std::vector<int> target;
    std::vector<int> pos_rem;
    std::vector<int> neg_rem;
    std::vector<int> queue;
    std::vector<int> row_seen;
    std::vector<int> row_saved;
    std::vector<int> var_trail;

    struct RowSnap {
        int r;
        int t;
        int p;
        int n;
    };

    std::vector<RowSnap> row_trail;
    int stamp = 1;

    explicit WorkerState(const Model& model) : M(&model) {
        x.assign(M->nvars, -1);
        target.assign(M->nrows, 0);
        pos_rem.assign(M->nrows, 0);
        neg_rem.assign(M->nrows, 0);
        row_seen.assign(M->nrows, 0);
        row_saved.assign(M->nrows, 0);

        for (int r = 0; r < M->nrows; ++r) {
            target[r] = M->rhs[r];
            int p = 0;
            int n = 0;
            for (int i = M->row_offsets[r]; i < M->row_offsets[r + 1]; ++i) {
                int c = M->all_coeffs[i];
                if (c > 0) p += c;
                else n += c;
            }
            pos_rem[r] = p;
            neg_rem[r] = n;
        }

        queue.reserve(M->nrows);
        var_trail.reserve(M->nvars);
        row_trail.reserve(M->nrows);
    }

    inline bool assign_var(int v, int val) {
        if (x[v] != -1) return x[v] == val;
        x[v] = static_cast<signed char>(val);
        var_trail.push_back(v);

        for (const auto& rc : M->var_adj[v]) {
            int r = rc.first;
            int c = rc.second;
            if (row_saved[r] != stamp) {
                row_saved[r] = stamp;
                row_trail.push_back({r, target[r], pos_rem[r], neg_rem[r]});
            }
            target[r] -= c * val;
            if (c > 0) pos_rem[r] -= c;
            else neg_rem[r] -= c;
            if (target[r] < neg_rem[r] || target[r] > pos_rem[r]) return false;
            if (row_seen[r] != stamp) {
                row_seen[r] = stamp;
                queue.push_back(r);
            }
        }
        const auto& implications = (val == 0) ? M->implications_when_zero[v] : M->implications_when_one[v];
        for (const auto& implied : implications) {
            if (!assign_var(implied.first, static_cast<int>(implied.second))) return false;
        }
        return true;
    }

    bool propagate() {
        std::size_t head = 0;
        while (head < queue.size()) {
            int r = queue[head++];
            int mc = M->row_max_coeff[r];
            for (int i = M->row_offsets[r]; i < M->row_offsets[r + 1]; ++i) {
                int v = M->all_vars[i];
                if (x[v] != -1) continue;
                int c = M->all_coeffs[i];
                int t = target[r];
                int p = pos_rem[r];
                int n = neg_rem[r];
                if (t > n + mc - 1 && t < p - mc + 1) continue;
                bool ok0 = (t >= n + (c < 0 ? -c : 0) && t <= p - (c > 0 ? c : 0));
                bool ok1 = (t - c >= n + (c < 0 ? -c : 0) && t - c <= p - (c > 0 ? c : 0));
                if (!ok0 && !ok1) return false;
                if (!ok0 || !ok1) {
                    if (!assign_var(v, ok1 ? 1 : 0)) return false;
                }
            }
        }
        return true;
    }

    void undo(std::size_t v_cp, std::size_t r_cp) {
        for (std::size_t i = var_trail.size(); i-- > v_cp;) x[var_trail[i]] = -1;
        var_trail.resize(v_cp);
        for (std::size_t i = row_trail.size(); i-- > r_cp;) {
            const auto& snap = row_trail[i];
            target[snap.r] = snap.t;
            pos_rem[snap.r] = snap.p;
            neg_rem[snap.r] = snap.n;
        }
        row_trail.resize(r_cp);
    }
};

struct DSU {
    std::vector<int> parent;
    std::vector<unsigned char> rank;
    std::vector<unsigned char> parity_to_parent;
    bool ok = true;

    explicit DSU(int n) : parent(n), rank(n, 0), parity_to_parent(n, 0) {
        std::iota(parent.begin(), parent.end(), 0);
    }

    std::pair<int, unsigned char> find(int x) {
        if (parent[x] == x) return {x, 0};
        auto root = find(parent[x]);
        parity_to_parent[x] ^= root.second;
        parent[x] = root.first;
        return {parent[x], parity_to_parent[x]};
    }

    bool unite(int a, int b, unsigned char xor_value) {
        auto fa = find(a);
        auto fb = find(b);
        if (fa.first == fb.first) {
            if ((fa.second ^ fb.second) != xor_value) ok = false;
            return ok;
        }

        if (rank[fa.first] < rank[fb.first]) std::swap(fa, fb);
        parent[fb.first] = fa.first;
        parity_to_parent[fb.first] = fa.second ^ fb.second ^ xor_value;
        if (rank[fa.first] == rank[fb.first]) ++rank[fa.first];
        return true;
    }
};

struct TLSData {
    uint64_t count = 0;
    uint64_t written = 0;
    uint64_t next = 1000000;
    std::chrono::steady_clock::time_point start;
    std::vector<uint64_t> pack;
    std::vector<char> io_buffer;
    std::unordered_map<uint64_t, uint64_t> mask_counts;
    std::ofstream shard;
};

static thread_local TLSData tls;

static std::string strip_spaces(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char ch : s) {
        if (!std::isspace(static_cast<unsigned char>(ch))) out.push_back(ch);
    }
    return out;
}

struct BoolAtom {
    bool is_const = false;
    int value = 0;
    int var = -1;
};

struct ParsedLine {
    std::vector<Row> rows;
    std::vector<std::pair<int, unsigned char>> fixed_assignments;
    std::vector<std::tuple<int, unsigned char, int, unsigned char>> implications;
};

static int parse_signed_int(const std::string& s, std::size_t& i) {
    bool neg = false;
    if (i < s.size() && (s[i] == '+' || s[i] == '-')) {
        neg = (s[i] == '-');
        ++i;
    }
    long long v = 0;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
        v = 10 * v + (s[i] - '0');
        ++i;
    }
    if (neg) v = -v;
    return static_cast<int>(v);
}

static BoolAtom parse_bool_atom(const std::string& s, int& max_var_seen) {
    if (s == "0" || s == "1") return BoolAtom{true, s[0] - '0', -1};
    if (s.size() >= 3 && s[0] == 'x' && s[1] == '_') {
        int var = std::stoi(s.substr(2)) - 1;
        max_var_seen = std::max(max_var_seen, var);
        return BoolAtom{false, 0, var};
    }
    throw std::runtime_error("Expected Boolean atom x_i or literal 0/1, got: " + s);
}

static Row parse_equality_row(const std::string& lhs, int rhs, int& max_var_seen) {
    Row row;
    row.rhs = rhs;
    std::unordered_map<int, int> merged;
    std::size_t i = 0;
    while (i < lhs.size()) {
        int sign = +1;
        while (i < lhs.size() && (lhs[i] == '+' || lhs[i] == '-')) {
            if (lhs[i] == '-') sign = -sign;
            ++i;
        }
        int coeff = 1;
        std::size_t c0 = i;
        while (i < lhs.size() && std::isdigit(static_cast<unsigned char>(lhs[i]))) ++i;
        if (i > c0) {
            coeff = std::stoi(lhs.substr(c0, i - c0));
            if (i < lhs.size() && lhs[i] == '*') ++i;
        }
        i += 2;
        std::size_t v0 = i;
        while (i < lhs.size() && std::isdigit(static_cast<unsigned char>(lhs[i]))) ++i;
        int var = std::stoi(lhs.substr(v0, i - v0)) - 1;
        max_var_seen = std::max(max_var_seen, var);
        merged[var] += sign * coeff;
    }

    std::vector<std::pair<int, int>> items;
    for (const auto& kv : merged) {
        if (kv.second != 0) items.push_back(kv);
    }
    std::sort(items.begin(), items.end());
    for (const auto& kv : items) {
        row.vars.push_back(kv.first);
        row.coeffs.push_back(kv.second);
    }
    return row;
}

static ParsedLine parse_line(const std::string& line, int& max_var_seen) {
    std::string s = strip_spaces(line);
    if (s.empty()) return ParsedLine{};

    ParsedLine out;
    auto neq = s.find("!=");
    auto leq = s.find("<=");
    auto geq = s.find(">=");
    if (neq != std::string::npos || leq != std::string::npos || geq != std::string::npos) {
        std::string op;
        std::size_t pos = std::string::npos;
        if (neq != std::string::npos) {
            op = "!=";
            pos = neq;
        } else if (leq != std::string::npos) {
            op = "<=";
            pos = leq;
        } else {
            op = ">=";
            pos = geq;
        }
        BoolAtom lhs = parse_bool_atom(s.substr(0, pos), max_var_seen);
        BoolAtom rhs = parse_bool_atom(s.substr(pos + 2), max_var_seen);

        if (op == ">=") std::swap(lhs, rhs);

        if (op == "!=") {
            if (lhs.is_const && rhs.is_const) {
                if (lhs.value == rhs.value) out.rows.push_back(Row{{}, {}, 1});
                return out;
            }
            if (lhs.is_const) {
                out.fixed_assignments.push_back({rhs.var, static_cast<unsigned char>(1 - lhs.value)});
                return out;
            }
            if (rhs.is_const) {
                out.fixed_assignments.push_back({lhs.var, static_cast<unsigned char>(1 - rhs.value)});
                return out;
            }
            Row row;
            row.vars = {lhs.var, rhs.var};
            row.coeffs = {1, 1};
            row.rhs = 1;
            out.rows.push_back(std::move(row));
            return out;
        }

        if (lhs.is_const && rhs.is_const) {
            if (lhs.value > rhs.value) out.rows.push_back(Row{{}, {}, 1});
            return out;
        }
        if (lhs.is_const) {
            if (lhs.value == 1) out.fixed_assignments.push_back({rhs.var, 1});
            return out;
        }
        if (rhs.is_const) {
            if (rhs.value == 0) out.fixed_assignments.push_back({lhs.var, 0});
            return out;
        }
        out.implications.push_back({lhs.var, 1, rhs.var, 1});
        out.implications.push_back({rhs.var, 0, lhs.var, 0});
        return out;
    }

    auto eq = s.find('=');
    if (eq == std::string::npos) throw std::runtime_error("Missing supported relation operator");
    std::string lhs = s.substr(0, eq);
    std::string rhs_s = s.substr(eq + 1);
    std::size_t rp = 0;
    int rhs = parse_signed_int(rhs_s, rp);
    out.rows.push_back(parse_equality_row(lhs, rhs, max_var_seen));
    return out;
}

static Model load_model(const std::string& filename) {
    std::ifstream fin(filename);
    if (!fin) throw std::runtime_error("Cannot open file: " + filename);

    Model model;
    std::string line;
    int max_var_seen = -1;
    std::vector<Row> rows;
    std::vector<std::pair<int, unsigned char>> fixed_assignments;
    std::vector<std::tuple<int, unsigned char, int, unsigned char>> implications;
    while (std::getline(fin, line)) {
        if (!strip_spaces(line).empty()) {
            ParsedLine parsed = parse_line(line, max_var_seen);
            rows.insert(rows.end(), parsed.rows.begin(), parsed.rows.end());
            fixed_assignments.insert(
                fixed_assignments.end(), parsed.fixed_assignments.begin(), parsed.fixed_assignments.end()
            );
            implications.insert(implications.end(), parsed.implications.begin(), parsed.implications.end());
        }
    }

    model.nvars = max_var_seen + 1;
    model.nrows = static_cast<int>(rows.size());
    model.var_adj.assign(model.nvars, {});
    model.implications_when_zero.assign(model.nvars, {});
    model.implications_when_one.assign(model.nvars, {});

    std::vector<int> score(model.nvars, 0);
    std::vector<int> degree(model.nvars, 0);
    model.row_offsets.push_back(0);

    for (int r = 0; r < model.nrows; ++r) {
        model.rhs.push_back(rows[r].rhs);
        int max_coeff = 0;
        for (std::size_t k = 0; k < rows[r].vars.size(); ++k) {
            int v = rows[r].vars[k];
            int c = rows[r].coeffs[k];
            model.all_vars.push_back(v);
            model.all_coeffs.push_back(c);
            model.var_adj[v].push_back({r, c});
            score[v] += std::abs(c);
            degree[v] += 1;
            max_coeff = std::max(max_coeff, std::abs(c));
        }
        model.row_max_coeff.push_back(max_coeff);
        model.row_offsets.push_back(static_cast<int>(model.all_vars.size()));
    }

    model.var_order.resize(model.nvars);
    for (int v = 0; v < model.nvars; ++v) model.var_order[v] = v;
    std::sort(model.var_order.begin(), model.var_order.end(), [&](int a, int b) {
        if (score[a] != score[b]) return score[a] > score[b];
        if (degree[a] != degree[b]) return degree[a] > degree[b];
        return a < b;
    });

    auto add_fixed = [&](int var, unsigned char value) {
        model.fixed_assignments.push_back({var, value});
    };
    for (const auto& fixed : fixed_assignments) add_fixed(fixed.first, fixed.second);

    for (const auto& implication : implications) {
        int src = std::get<0>(implication);
        unsigned char src_val = std::get<1>(implication);
        int dst = std::get<2>(implication);
        unsigned char dst_val = std::get<3>(implication);
        auto& bucket = (src_val == 0) ? model.implications_when_zero[src] : model.implications_when_one[src];
        bucket.push_back({dst, dst_val});
    }

    return model;
}

static std::string default_metadata_path(const std::string& out_file) {
    std::filesystem::path path(out_file);
    path.replace_extension(".json");
    return path.string();
}

static PropertyInput parse_property_input(const std::string& spec) {
    auto pos = spec.find('=');
    PropertyInput input;
    if (pos == std::string::npos) {
        input.path = spec;
        input.name = std::filesystem::path(spec).stem().string();
    } else {
        input.name = spec.substr(0, pos);
        input.path = spec.substr(pos + 1);
    }
    if (input.path.empty()) throw std::runtime_error("Bad --property specification: " + spec);
    if (input.name.empty()) input.name = std::filesystem::path(input.path).stem().string();
    return input;
}

static Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--base-property" && i + 1 < argc) {
            opt.base_property = argv[++i];
        } else if (a == "--property" && i + 1 < argc) {
            opt.properties.push_back(parse_property_input(argv[++i]));
        } else if (a == "--threads" && i + 1 < argc) {
            opt.threads = std::max(1, std::stoi(argv[++i]));
        } else if (a == "--spawn-depth" && i + 1 < argc) {
            opt.spawn_depth = std::max(0, std::stoi(argv[++i]));
        } else if (a == "--report-every" && i + 1 < argc) {
            opt.report_every = std::max<uint64_t>(1, std::stoull(argv[++i]));
        } else if (a == "--count-only") {
            opt.count_only = true;
        } else if (a == "--write-nonzero-masks-only") {
            opt.write_nonzero_masks_only = true;
        } else if (a == "--out-file" && i + 1 < argc) {
            opt.out_file = argv[++i];
        } else if (a == "--metadata-file" && i + 1 < argc) {
            opt.metadata_file = argv[++i];
        } else if (!a.empty() && a[0] != '-' && opt.base_property.empty()) {
            opt.base_property = a;
        } else {
            throw std::runtime_error(
                "Usage:\n"
                "  solve_omp_opt --base-property BASE.txt [--property name=PATH]...\n"
                "                [--threads N] [--spawn-depth D] [--report-every K]\n"
                "                [--count-only] [--out-file results.bin]\n"
                "                [--write-nonzero-masks-only]\n"
                "                [--metadata-file results.json]\n"
                "\n"
                "The first positional argument may also be used as --base-property."
            );
        }
    }

    if (opt.base_property.empty()) {
        throw std::runtime_error("Missing --base-property");
    }
    if (opt.properties.size() > 64) {
        throw std::runtime_error("At most 64 additional properties are supported");
    }
    if (!opt.out_file.empty() && opt.metadata_file.empty()) {
        opt.metadata_file = default_metadata_path(opt.out_file);
    }
    return opt;
}

static bool try_absorb_affine_row(const Row& row, DSU& dsu) {
    if (row.vars.empty()) {
        if (row.rhs != 0) dsu.ok = false;
        return true;
    }

    std::vector<int> coeffs = row.coeffs;
    int rhs = row.rhs;
    if (coeffs[0] < 0) {
        for (int& coeff : coeffs) coeff = -coeff;
        rhs = -rhs;
    }

    if (coeffs.size() == 1) {
        int a = coeffs[0];
        if (a == 0) {
            if (rhs != 0) dsu.ok = false;
            return true;
        }
        if (rhs % a != 0) return false;
        int value = rhs / a;
        if (value < 0 || value > 1) return false;
        dsu.unite(row.vars[0] + 1, 0, static_cast<unsigned char>(value));
        return true;
    }

    if (coeffs.size() == 2) {
        int a = coeffs[0];
        int b = coeffs[1];
        if (a == -b && rhs == 0) {
            dsu.unite(row.vars[0] + 1, row.vars[1] + 1, 0);
            return true;
        }
        if (a == b && rhs == a) {
            dsu.unite(row.vars[0] + 1, row.vars[1] + 1, 1);
            return true;
        }
    }

    return false;
}

static std::string escape_json(const std::string& s) {
    std::ostringstream out;
    for (char ch : s) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    out << "\\u"
                        << std::hex << std::uppercase
                        << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(ch))
                        << std::dec << std::nouppercase;
                } else {
                    out << ch;
                }
        }
    }
    return out.str();
}

static CompiledProperty compile_property(const PropertyInput& input, int expected_nvars) {
    Model model = load_model(input.path);
    if (model.nvars > expected_nvars) {
        throw std::runtime_error(
            "Property variable count mismatch for " + input.path + ": expected " +
            std::to_string(expected_nvars) + " or fewer, got " + std::to_string(model.nvars)
        );
    }

    CompiledProperty property;
    property.name = input.name;
    property.path = input.path;
    property.nrows = model.nrows;
    for (const auto& fixed : model.fixed_assignments) {
        property.fixed_checks.push_back({fixed.first, fixed.second});
    }
    for (int v = 0; v < model.nvars; ++v) {
        for (const auto& implied : model.implications_when_one[v]) {
            if (implied.second == 1) property.implication_checks.push_back({v, implied.first});
        }
        for (const auto& implied : model.implications_when_zero[v]) {
            if (implied.second == 0) property.implication_checks.push_back({implied.first, v});
        }
    }

    DSU dsu(expected_nvars + 1);
    for (int r = 0; r < model.nrows; ++r) {
        Row row;
        row.rhs = model.rhs[r];
        for (int i = model.row_offsets[r]; i < model.row_offsets[r + 1]; ++i) {
            row.vars.push_back(model.all_vars[i]);
            row.coeffs.push_back(model.all_coeffs[i]);
        }
        if (!try_absorb_affine_row(row, dsu)) property.generic_rows.push_back(std::move(row));
    }

    if (!dsu.ok) {
        property.always_false = true;
        property.generic_rows.clear();
        return property;
    }

    const auto const_info = dsu.find(0);
    const int const_root = const_info.first;
    const unsigned char const_parity_to_root = const_info.second;
    std::unordered_map<int, int> anchor_by_root;
    std::unordered_map<int, unsigned char> anchor_parity;

    for (int v = 0; v < expected_nvars; ++v) {
        auto fv = dsu.find(v + 1);
        int root = fv.first;
        unsigned char parity_to_root = fv.second;

        auto it = anchor_by_root.find(root);
        if (it == anchor_by_root.end()) {
            anchor_by_root[root] = v;
            anchor_parity[root] = parity_to_root;
            if (root == const_root) {
                // DSU parity is recorded relative to the component root, not relative to the
                // distinguished constant node 0. If node 0 is not itself the root, we need to
                // translate through the constant node's parity to recover the actual fixed value.
                property.fixed_checks.push_back({
                    v,
                    static_cast<unsigned char>(parity_to_root ^ const_parity_to_root),
                });
            }
            continue;
        }

        unsigned char xor_value = static_cast<unsigned char>(anchor_parity[root] ^ parity_to_root);
        property.relation_checks.push_back({it->second, v, xor_value});
    }

    return property;
}

static uint64_t classify_solution(const WorkerState& state, const Runtime& rt) {
    uint64_t mask = 0;
    for (std::size_t i = 0; i < rt.properties.size(); ++i) {
        if (rt.properties[i].check(state.x)) mask |= (uint64_t(1) << i);
    }
    return mask;
}

static void flush_thread_output() {
    if (!tls.shard.is_open() || tls.io_buffer.empty()) return;
    tls.shard.write(tls.io_buffer.data(), static_cast<std::streamsize>(tls.io_buffer.size()));
    tls.io_buffer.clear();
}

static void append_mask_bytes(uint64_t mask, std::size_t mask_bytes) {
    for (std::size_t i = 0; i < mask_bytes; ++i) {
        tls.io_buffer.push_back(static_cast<char>((mask >> (8 * i)) & 0xFF));
    }
}

static void record_solution(const WorkerState& state, Runtime& rt) {
    ++tls.count;
    uint64_t mask = classify_solution(state, rt);
    ++tls.mask_counts[mask];

    bool should_write = rt.write_output && (!rt.write_nonzero_masks_only || mask != 0);
    if (should_write) {
        ++tls.written;
        std::fill(tls.pack.begin(), tls.pack.end(), 0);
        for (int v = 0; v < state.M->nvars; ++v) {
            if (state.x[v] == 1) tls.pack[v >> 6] |= (uint64_t(1) << (v & 63));
        }
        append_mask_bytes(mask, rt.mask_bytes);
        const char* bytes = reinterpret_cast<const char*>(tls.pack.data());
        tls.io_buffer.insert(tls.io_buffer.end(), bytes, bytes + rt.solution_bytes);
        if (tls.io_buffer.size() >= 1 << 20) flush_thread_output();
    }

    if (tls.count >= tls.next) {
        auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - tls.start).count();
        std::cerr << "[thread " << omp_get_thread_num() << "] count=" << tls.count
                  << " rate=" << (elapsed > 0 ? tls.count / elapsed : 0.0) << " sol/s" << std::endl;
        tls.next += rt.report_every;
    }
}

static void dfs_seq(WorkerState& state, int v_idx, Runtime& rt) {
    while (v_idx < state.M->nvars && state.x[state.M->var_order[v_idx]] != -1) ++v_idx;
    if (v_idx == state.M->nvars) {
        record_solution(state, rt);
        return;
    }

    int v = state.M->var_order[v_idx];
    for (int val = 0; val <= 1; ++val) {
        ++state.stamp;
        state.queue.clear();
        std::size_t v_cp = state.var_trail.size();
        std::size_t r_cp = state.row_trail.size();
        if (state.assign_var(v, val) && state.propagate()) dfs_seq(state, v_idx + 1, rt);
        state.undo(v_cp, r_cp);
    }
}

static void dfs_task(WorkerState state, int v_idx, int depth, Runtime& rt) {
    while (v_idx < state.M->nvars && state.x[state.M->var_order[v_idx]] != -1) ++v_idx;
    if (v_idx == state.M->nvars) {
        record_solution(state, rt);
        return;
    }
    if (depth <= 0) {
        dfs_seq(state, v_idx, rt);
        return;
    }

    int v = state.M->var_order[v_idx];
    for (int val = 0; val <= 1; ++val) {
        WorkerState child = state;
        ++child.stamp;
        child.queue.clear();
        if (child.assign_var(v, val) && child.propagate()) {
            #pragma omp task default(none) firstprivate(child, v_idx, depth) shared(rt)
            dfs_task(child, v_idx + 1, depth - 1, rt);
        }
    }
    #pragma omp taskwait
}

struct OutputHeader {
    char magic[4];
    uint32_t version;
    uint32_t nvars;
    uint32_t nproperties;
    uint32_t mask_bytes;
    uint32_t solution_bytes;
    uint64_t total_records;
};

static void write_final_output(const Runtime& rt, uint64_t total_records) {
    if (!rt.write_output) return;

    std::filesystem::path out_path(rt.out_file);
    if (!out_path.parent_path().empty()) {
        std::filesystem::create_directories(out_path.parent_path());
    }

    std::ofstream out(rt.out_file, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("Cannot open output file: " + rt.out_file);

    OutputHeader header{{'C', 'P', 'M', '1'},
                        1u,
                        static_cast<uint32_t>(rt.nvars),
                        static_cast<uint32_t>(rt.properties.size()),
                        static_cast<uint32_t>(rt.mask_bytes),
                        static_cast<uint32_t>(rt.solution_bytes),
                        total_records};
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));

    std::vector<char> buffer(1 << 20);
    for (const auto& shard : rt.shard_files) {
        std::ifstream in(shard, std::ios::binary);
        if (!in) throw std::runtime_error("Cannot open shard file: " + shard);
        while (in) {
            in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
            std::streamsize got = in.gcount();
            if (got > 0) out.write(buffer.data(), got);
        }
        in.close();
        std::filesystem::remove(shard);
    }
}

static std::vector<uint64_t> compute_property_counts(
    const std::vector<CompiledProperty>& properties,
    const std::unordered_map<uint64_t, uint64_t>& exact_counts
) {
    std::vector<uint64_t> counts(properties.size(), 0);
    for (const auto& kv : exact_counts) {
        uint64_t mask = kv.first;
        uint64_t count = kv.second;
        for (std::size_t i = 0; i < properties.size(); ++i) {
            if (mask & (uint64_t(1) << i)) counts[i] += count;
        }
    }
    return counts;
}

static void write_metadata_json(
    const Options& opt,
    const Model& base,
    const Runtime& rt,
    uint64_t total_solutions,
    uint64_t written_solutions,
    double elapsed_seconds,
    const std::unordered_map<uint64_t, uint64_t>& exact_counts
) {
    if (opt.metadata_file.empty()) return;

    std::ofstream out(opt.metadata_file, std::ios::trunc);
    if (!out) throw std::runtime_error("Cannot open metadata file: " + opt.metadata_file);

    auto property_counts = compute_property_counts(rt.properties, exact_counts);
    std::vector<std::pair<uint64_t, uint64_t>> exact_sorted(exact_counts.begin(), exact_counts.end());
    std::sort(exact_sorted.begin(), exact_sorted.end());

    out << "{\n";
    out << "  \"format\": \"ca_property_mask_v1\",\n";
    out << "  \"base_property\": {\n";
    out << "    \"path\": \"" << escape_json(opt.base_property) << "\",\n";
    out << "    \"nvars\": " << base.nvars << ",\n";
    out << "    \"nrows\": " << base.nrows << "\n";
    out << "  },\n";
    out << "  \"additional_properties\": [\n";
    for (std::size_t i = 0; i < rt.properties.size(); ++i) {
        const auto& prop = rt.properties[i];
        out << "    {\n";
        out << "      \"bit\": " << i << ",\n";
        out << "      \"name\": \"" << escape_json(prop.name) << "\",\n";
        out << "      \"path\": \"" << escape_json(prop.path) << "\",\n";
        out << "      \"nrows\": " << prop.nrows << ",\n";
        out << "      \"fixed_checks\": " << prop.fixed_checks.size() << ",\n";
        out << "      \"relation_checks\": " << prop.relation_checks.size() << ",\n";
        out << "      \"implication_checks\": " << prop.implication_checks.size() << ",\n";
        out << "      \"generic_rows\": " << prop.generic_rows.size() << "\n";
        out << "    }" << (i + 1 == rt.properties.size() ? "\n" : ",\n");
    }
    out << "  ],\n";
    out << "  \"mask_bits_are_little_endian\": true,\n";
    out << "  \"mask_bytes_per_record\": " << rt.mask_bytes << ",\n";
    out << "  \"solution_bytes_per_record\": " << rt.solution_bytes << ",\n";
    out << "  \"write_nonzero_masks_only\": " << (rt.write_nonzero_masks_only ? "true" : "false") << ",\n";
    out << "  \"total_base_solutions\": " << total_solutions << ",\n";
    out << "  \"written_solution_records\": " << written_solutions << ",\n";
    out << "  \"elapsed_seconds\": " << elapsed_seconds << ",\n";
    out << "  \"property_counts\": [\n";
    for (std::size_t i = 0; i < rt.properties.size(); ++i) {
        out << "    {\n";
        out << "      \"bit\": " << i << ",\n";
        out << "      \"name\": \"" << escape_json(rt.properties[i].name) << "\",\n";
        out << "      \"count\": " << property_counts[i] << "\n";
        out << "    }" << (i + 1 == rt.properties.size() ? "\n" : ",\n");
    }
    out << "  ],\n";
    out << "  \"exact_mask_counts\": [\n";
    for (std::size_t i = 0; i < exact_sorted.size(); ++i) {
        uint64_t mask = exact_sorted[i].first;
        uint64_t count = exact_sorted[i].second;
        out << "    {\n";
        out << "      \"mask\": " << mask << ",\n";
        out << "      \"mask_hex\": \"0x" << std::hex << mask << std::dec << "\",\n";
        out << "      \"count\": " << count << "\n";
        out << "    }" << (i + 1 == exact_sorted.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
}

int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);
        Model base = load_model(opt.base_property);

        Runtime rt;
        rt.report_every = opt.report_every;
        rt.write_nonzero_masks_only = opt.write_nonzero_masks_only;
        rt.nvars = base.nvars;
        rt.solution_bytes = ((base.nvars + 63) / 64) * sizeof(uint64_t);
        rt.mask_bytes = (opt.properties.size() + 7) / 8;
        rt.record_bytes = rt.mask_bytes + rt.solution_bytes;
        rt.out_file = opt.out_file;
        rt.metadata_file = opt.metadata_file;
        rt.write_output = !opt.count_only && !opt.out_file.empty();

        rt.properties.reserve(opt.properties.size());
        for (const auto& input : opt.properties) {
            rt.properties.push_back(compile_property(input, base.nvars));
        }

        if (rt.write_output) {
            std::filesystem::path out_path(rt.out_file);
            if (!out_path.parent_path().empty()) {
                std::filesystem::create_directories(out_path.parent_path());
            }
            rt.shard_files.resize(opt.threads);
            for (int tid = 0; tid < opt.threads; ++tid) {
                rt.shard_files[tid] = rt.out_file + ".part." + std::to_string(tid);
            }
        }

        std::vector<uint64_t> totals(opt.threads, 0);
        std::vector<uint64_t> written_totals(opt.threads, 0);
        std::vector<std::unordered_map<uint64_t, uint64_t>> exact_by_thread(opt.threads);
        auto start = std::chrono::steady_clock::now();

        omp_set_num_threads(opt.threads);
        #pragma omp parallel default(none) shared(base, rt, opt, totals, written_totals, exact_by_thread)
        {
            int tid = omp_get_thread_num();
            tls.count = 0;
            tls.written = 0;
            tls.next = rt.report_every;
            tls.start = std::chrono::steady_clock::now();
            tls.pack.assign((base.nvars + 63) / 64, 0);
            tls.io_buffer.clear();
            tls.mask_counts.clear();
            if (rt.write_output) {
                tls.shard.open(rt.shard_files[tid], std::ios::binary | std::ios::trunc);
                if (!tls.shard) throw std::runtime_error("Cannot open shard file");
            }

            #pragma omp single
            {
                WorkerState root(base);
                ++root.stamp;
                root.queue.clear();
                bool ok = true;
                for (const auto& fixed : base.fixed_assignments) {
                    if (!root.assign_var(fixed.first, static_cast<int>(fixed.second))) {
                        ok = false;
                        break;
                    }
                }
                for (int r = 0; ok && r < base.nrows; ++r) {
                    if (root.row_seen[r] != root.stamp) {
                        root.row_seen[r] = root.stamp;
                        root.queue.push_back(r);
                    }
                }
                if (ok && root.propagate()) dfs_task(root, 0, opt.spawn_depth, rt);
            }

            flush_thread_output();
            if (tls.shard.is_open()) tls.shard.close();
            totals[tid] = tls.count;
            written_totals[tid] = tls.written;
            exact_by_thread[tid] = std::move(tls.mask_counts);
        }

        uint64_t total = 0;
        for (uint64_t value : totals) total += value;
        uint64_t written_total = 0;
        for (uint64_t value : written_totals) written_total += value;

        std::unordered_map<uint64_t, uint64_t> exact_counts;
        for (auto& local : exact_by_thread) {
            for (const auto& kv : local) exact_counts[kv.first] += kv.second;
        }

        auto end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        if (rt.write_output) write_final_output(rt, written_total);
        write_metadata_json(opt, base, rt, total, written_total, elapsed, exact_counts);

        std::cout << "Total base-property solutions: " << total
                  << "\nElapsed: " << elapsed << " s"
                  << "\nAverage: " << (elapsed > 0 ? total / elapsed : 0.0) << " sol/s"
                  << std::endl;

        if (!rt.properties.empty()) {
            auto property_counts = compute_property_counts(rt.properties, exact_counts);
            std::cout << "\nAdditional property counts:" << std::endl;
            for (std::size_t i = 0; i < rt.properties.size(); ++i) {
                std::cout << "  [" << i << "] " << rt.properties[i].name
                          << ": " << property_counts[i] << std::endl;
            }
            std::vector<std::pair<uint64_t, uint64_t>> exact_sorted(exact_counts.begin(), exact_counts.end());
            std::sort(exact_sorted.begin(), exact_sorted.end());
            std::cout << "\nExact mask counts:" << std::endl;
            for (const auto& kv : exact_sorted) {
                std::cout << "  mask=0x" << std::hex << kv.first << std::dec
                          << " count=" << kv.second << std::endl;
            }
        }

        if (rt.write_output) {
            std::cout << "\nOutput file: " << rt.out_file << std::endl;
            std::cout << "Written solution records: " << written_total << std::endl;
        }
        if (!opt.metadata_file.empty()) {
            std::cout << "Metadata file: " << opt.metadata_file << std::endl;
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
}
