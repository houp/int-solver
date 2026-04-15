// Solver implemented by ChatGPT (GPT-5.4 Thinking) - full human review have NOT yet been done.
// Code evaluated and tested by Google Gemini-3 Flash Preview via Gemini CLI

// Needs C++17 or newer and OpenMP
// Tested on MacOS and Linux

#include <omp.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
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
    std::vector<Row> rows;
    std::vector<std::vector<std::pair<int,int>>> var_adj; // var -> (row, coeff)
    std::vector<int> var_order;
};

static std::string strip_spaces(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char ch : s) {
        if (!std::isspace(static_cast<unsigned char>(ch))) out.push_back(ch);
    }
    return out;
}

static int parse_signed_int(const std::string& s, std::size_t& i) {
    bool neg = false;
    if (i < s.size() && (s[i] == '+' || s[i] == '-')) {
        neg = (s[i] == '-');
        ++i;
    }
    if (i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i]))) {
        throw std::runtime_error("Expected integer near: " + s.substr(i));
    }
    long long v = 0;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
        v = 10 * v + (s[i] - '0');
        ++i;
    }
    if (neg) v = -v;
    if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max()) {
        throw std::runtime_error("Integer overflow");
    }
    return static_cast<int>(v);
}

static Row parse_line(const std::string& line, int& max_var_seen) {
    std::string s = strip_spaces(line);
    if (s.empty()) return Row{};

    auto eq = s.find('=');
    if (eq == std::string::npos) {
        throw std::runtime_error("Missing '=' in line: " + line);
    }

    std::string lhs = s.substr(0, eq);
    std::string rhs_s = s.substr(eq + 1);
    if (rhs_s.empty()) {
        throw std::runtime_error("Missing RHS in line: " + line);
    }

    std::size_t rp = 0;
    int rhs = parse_signed_int(rhs_s, rp);
    if (rp != rhs_s.size()) {
        throw std::runtime_error("Bad RHS in line: " + line);
    }

    std::unordered_map<int,int> merged;
    std::size_t i = 0;

    while (i < lhs.size()) {
        int sign = +1;

        // Accept repeated unary signs: +-, --, -+, etc.
        while (i < lhs.size() && (lhs[i] == '+' || lhs[i] == '-')) {
            if (lhs[i] == '-') sign = -sign;
            ++i;
        }

        int coeff = 1;

        // Optional explicit coefficient, e.g. 2*x_7 or 3x_7
        std::size_t c0 = i;
        while (i < lhs.size() && std::isdigit(static_cast<unsigned char>(lhs[i]))) {
            ++i;
        }
        if (i > c0) {
            coeff = std::stoi(lhs.substr(c0, i - c0));
            if (i < lhs.size() && lhs[i] == '*') ++i;
        }

        if (i + 1 >= lhs.size() || lhs[i] != 'x' || lhs[i + 1] != '_') {
            throw std::runtime_error("Expected variable x_k in line: " + line);
        }

        i += 2;
        std::size_t v0 = i;
        while (i < lhs.size() && std::isdigit(static_cast<unsigned char>(lhs[i]))) {
            ++i;
        }
        if (i == v0) {
            throw std::runtime_error("Bad variable token in line: " + line);
        }

        int var = std::stoi(lhs.substr(v0, i - v0)) - 1;
        if (var < 0) {
            throw std::runtime_error("Variable indices must start from 1");
        }

        max_var_seen = std::max(max_var_seen, var);
        merged[var] += sign * coeff;
    }

    Row row;
    row.rhs = rhs;

    std::vector<std::pair<int,int>> items;
    items.reserve(merged.size());
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

static Model load_model(const std::string& filename) {
    std::ifstream fin(filename);
    if (!fin) throw std::runtime_error("Cannot open file: " + filename);

    Model M;
    std::string line;
    int max_var_seen = -1;

    while (std::getline(fin, line)) {
        if (strip_spaces(line).empty()) continue;
        Row r = parse_line(line, max_var_seen);
        if (!r.vars.empty() || r.rhs != 0) {
            M.rows.push_back(std::move(r));
        }
    }

    M.nvars = max_var_seen + 1;
    M.var_adj.assign(M.nvars, {});
    std::vector<int> score(M.nvars, 0);
    std::vector<int> degree(M.nvars, 0);

    for (int r = 0; r < (int)M.rows.size(); ++r) {
        for (int k = 0; k < (int)M.rows[r].vars.size(); ++k) {
            int v = M.rows[r].vars[k];
            int c = M.rows[r].coeffs[k];
            M.var_adj[v].push_back({r, c});
            score[v] += std::abs(c);
            degree[v] += 1;
        }
    }

    M.var_order.resize(M.nvars);
    for (int v = 0; v < M.nvars; ++v) M.var_order[v] = v;
    std::sort(M.var_order.begin(), M.var_order.end(), [&](int a, int b) {
        if (score[a] != score[b]) return score[a] > score[b];
        if (degree[a] != degree[b]) return degree[a] > degree[b];
        return a < b;
    });

    return M;
}

struct WorkerState {
    const Model* M = nullptr;

    std::vector<signed char> x; // -1 unknown, 0, 1
    std::vector<int> target;    // rhs - assigned contribution
    std::vector<int> pos_rem;   // sum of positive coeffs of unassigned vars
    std::vector<int> neg_rem;   // sum of negative coeffs of unassigned vars

    struct RowSnap {
        int row;
        int target;
        int pos_rem;
        int neg_rem;
    };

    std::vector<RowSnap> row_trail;
    std::vector<int> var_trail;
    std::vector<int> queue;
    std::vector<int> row_seen;
    std::vector<int> row_saved;
    int stamp = 1;

    WorkerState() = default;

    explicit WorkerState(const Model& model) : M(&model) {
        int m = (int)M->rows.size();
        x.assign(M->nvars, -1);
        target.assign(m, 0);
        pos_rem.assign(m, 0);
        neg_rem.assign(m, 0);
        row_seen.assign(m, 0);
        row_saved.assign(m, 0);

        for (int r = 0; r < m; ++r) {
            target[r] = M->rows[r].rhs;
            int p = 0, n = 0;
            for (int c : M->rows[r].coeffs) {
                if (c > 0) p += c;
                else n += c;
            }
            pos_rem[r] = p;
            neg_rem[r] = n;
        }
    }

    inline bool feasible_row(int r) const {
        return neg_rem[r] <= target[r] && target[r] <= pos_rem[r];
    }

    inline void save_row_once(int r) {
        if (row_saved[r] != stamp) {
            row_saved[r] = stamp;
            row_trail.push_back({r, target[r], pos_rem[r], neg_rem[r]});
        }
    }

    inline void queue_row_once(int r) {
        if (row_seen[r] != stamp) {
            row_seen[r] = stamp;
            queue.push_back(r);
        }
    }

    void queue_all_rows_once() {
        for (int r = 0; r < (int)M->rows.size(); ++r) {
            queue_row_once(r);
        }
    }

    bool assign_var(int v, int value) {
        if (x[v] != -1) return x[v] == value;

        x[v] = (signed char)value;
        var_trail.push_back(v);

        for (const auto& rc : M->var_adj[v]) {
            int r = rc.first;
            int c = rc.second;

            save_row_once(r);

            target[r] -= c * value;
            if (c > 0) pos_rem[r] -= c;
            else neg_rem[r] -= c;

            if (!feasible_row(r)) return false;
            queue_row_once(r);
        }

        return true;
    }

    inline bool feasible_after(int r, int coeff, int value) const {
        int t = target[r] - coeff * value;
        int p = pos_rem[r] - (coeff > 0 ? coeff : 0);
        int n = neg_rem[r] - (coeff < 0 ? coeff : 0);
        return n <= t && t <= p;
    }

    bool propagate() {
        std::size_t head = 0;
        while (head < queue.size()) {
            int r = queue[head++];
            const Row& row = M->rows[r];

            for (int k = 0; k < (int)row.vars.size(); ++k) {
                int v = row.vars[k];
                if (x[v] != -1) continue;

                int c = row.coeffs[k];
                bool ok0 = feasible_after(r, c, 0);
                bool ok1 = feasible_after(r, c, 1);

                if (!ok0 && !ok1) return false;

                if (ok0 ^ ok1) {
                    if (!assign_var(v, ok1 ? 1 : 0)) return false;
                }
            }
        }
        return true;
    }

    void undo(std::size_t var_cp, std::size_t row_cp) {
        for (std::size_t i = var_trail.size(); i-- > var_cp; ) {
            x[var_trail[i]] = -1;
        }
        var_trail.resize(var_cp);

        for (std::size_t i = row_trail.size(); i-- > row_cp; ) {
            const RowSnap& s = row_trail[i];
            target[s.row] = s.target;
            pos_rem[s.row] = s.pos_rem;
            neg_rem[s.row] = s.neg_rem;
        }
        row_trail.resize(row_cp);
    }

    int choose_var() const {
        for (int v : M->var_order) {
            if (x[v] == -1) return v;
        }
        return -1;
    }
};

struct Runtime {
    bool write_output = false;
    uint64_t report_every = 10000;
    int nvars = 0;
    std::vector<std::ofstream> out_files;
};

struct TLSData {
    uint64_t local_solutions = 0;
    uint64_t next_report = 0;
    std::chrono::steady_clock::time_point start;
    std::vector<std::uint64_t> packbuf;
};

static thread_local TLSData tls;

static inline void tls_reset(uint64_t report_every, int nvars) {
    tls.local_solutions = 0;
    tls.next_report = report_every;
    tls.start = std::chrono::steady_clock::now();
    tls.packbuf.assign((nvars + 63) / 64, 0);
}

static inline void write_solution_binary(const WorkerState& S, Runtime& rt) {
    if (!rt.write_output) return;

    int tid = omp_get_thread_num();
    std::ofstream& fout = rt.out_files[tid];

    std::fill(tls.packbuf.begin(), tls.packbuf.end(), 0);
    for (int v = 0; v < S.M->nvars; ++v) {
        if (S.x[v] == 1) {
            tls.packbuf[(std::size_t)v >> 6] |= (std::uint64_t(1) << (v & 63));
        }
    }

    fout.write(reinterpret_cast<const char*>(tls.packbuf.data()),
               (std::streamsize)(tls.packbuf.size() * sizeof(std::uint64_t)));
}

static inline void record_solution(const WorkerState& S, Runtime& rt) {
    ++tls.local_solutions;
    write_solution_binary(S, rt);

    if (tls.local_solutions >= tls.next_report) {
        auto now = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(now - tls.start).count();
        double rate = sec > 0.0 ? (tls.local_solutions / sec) : 0.0;

        std::cerr
            << "[thread " << omp_get_thread_num() << "] "
            << "solutions=" << tls.local_solutions
            << " elapsed=" << sec << "s"
            << " rate=" << rate << " sol/s"
            << std::endl;

        tls.next_report += rt.report_every;
    }
}

static void dfs_seq(WorkerState& S, Runtime& rt) {
    int v = S.choose_var();
    if (v < 0) {
        record_solution(S, rt);
        return;
    }

    for (int value = 0; value <= 1; ++value) {
        ++S.stamp;
        S.queue.clear();

        std::size_t var_cp = S.var_trail.size();
        std::size_t row_cp = S.row_trail.size();

        if (S.assign_var(v, value) && S.propagate()) {
            dfs_seq(S, rt);
        }

        S.undo(var_cp, row_cp);
    }
}

// Parallel top-of-tree search using OpenMP tasks.
// State is copied only at task levels; below cutoff we switch to sequential DFS.
static void dfs_task(WorkerState S, int spawn_depth, Runtime& rt) {
    int v = S.choose_var();
    if (v < 0) {
        record_solution(S, rt);
        return;
    }

    if (spawn_depth <= 0) {
        dfs_seq(S, rt);
        return;
    }

    for (int value = 0; value <= 1; ++value) {
        WorkerState child = S;

        ++child.stamp;
        child.queue.clear();

        if (child.assign_var(v, value) && child.propagate()) {
            #pragma omp task default(none) firstprivate(child, spawn_depth) shared(rt)
            {
                dfs_task(child, spawn_depth - 1, rt);
            }
        }
    }

    #pragma omp taskwait
}

struct Options {
    std::string input_file;
    int threads = std::max(1u, std::thread::hardware_concurrency());
    int spawn_depth = 8;
    uint64_t report_every = 10000;
    bool count_only = false;
    std::string out_dir;
};

static Options parse_args(int argc, char** argv) {
    if (argc < 2) {
        throw std::runtime_error(
            "Usage:\n"
            "  solver_omp equations.txt [--threads N] [--spawn-depth D]\n"
            "                           [--report-every K] [--count-only]\n"
            "                           [--out-dir solutions_bin]\n"
        );
    }

    Options opt;
    opt.input_file = argv[1];

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--threads" && i + 1 < argc) {
            opt.threads = std::max(1, std::stoi(argv[++i]));
        } else if (a == "--spawn-depth" && i + 1 < argc) {
            opt.spawn_depth = std::max(0, std::stoi(argv[++i]));
        } else if (a == "--report-every" && i + 1 < argc) {
            opt.report_every = std::max<uint64_t>(1, std::stoull(argv[++i]));
        } else if (a == "--count-only") {
            opt.count_only = true;
        } else if (a == "--out-dir" && i + 1 < argc) {
            opt.out_dir = argv[++i];
        } else {
            throw std::runtime_error("Unknown argument: " + a);
        }
    }

    return opt;
}

int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);
        Model M = load_model(opt.input_file);

        Runtime rt;
        rt.write_output = (!opt.count_only && !opt.out_dir.empty());
        rt.report_every = opt.report_every;
        rt.nvars = M.nvars;

        if (rt.write_output) {
            std::filesystem::create_directories(opt.out_dir);
            rt.out_files.resize((std::size_t)opt.threads);
            for (int t = 0; t < opt.threads; ++t) {
                std::ostringstream oss;
                oss << opt.out_dir << "/worker_" << t << ".bin";
                rt.out_files[(std::size_t)t].open(oss.str(), std::ios::binary);
                if (!rt.out_files[(std::size_t)t]) {
                    throw std::runtime_error("Cannot open output file: " + oss.str());
                }
            }
        }

        std::cerr << "Rows: " << M.rows.size() << "\n";
        std::cerr << "Variables: x_1 .. x_" << M.nvars << "\n";
        std::cerr << "Threads: " << opt.threads << "\n";
        std::cerr << "Spawn depth: " << opt.spawn_depth << "\n";
        std::cerr << "Report every: " << opt.report_every << "\n";

        auto global_start = std::chrono::steady_clock::now();
        std::vector<uint64_t> thread_totals((std::size_t)opt.threads, 0);

        omp_set_dynamic(0);
        omp_set_num_threads(opt.threads);

        #pragma omp parallel default(none) shared(M, rt, opt, thread_totals)
        {
            tls_reset(opt.report_every, M.nvars);

            #pragma omp single
            {
                WorkerState root(M);

                // Initial propagation from all rows once.
                ++root.stamp;
                root.queue.clear();
                root.queue_all_rows_once();

                if (root.propagate()) {
                    dfs_task(root, opt.spawn_depth, rt);
                }
            }

            thread_totals[(std::size_t)omp_get_thread_num()] = tls.local_solutions;
        }

        uint64_t total = 0;
        for (uint64_t v : thread_totals) total += v;

        auto global_end = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(global_end - global_start).count();
        double rate = sec > 0.0 ? (total / sec) : 0.0;

        std::cout << "Total solutions: " << total << "\n";
        std::cout << "Elapsed: " << sec << " s\n";
        std::cout << "Average global rate: " << rate << " sol/s\n";

        if (rt.write_output) {
            std::cout << "Binary solutions written to: " << opt.out_dir << "\n";
            std::cout << "Each solution uses "
                      << ((M.nvars + 63) / 64) * 8
                      << " bytes.\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}