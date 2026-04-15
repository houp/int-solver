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
#include <memory>
#include <cstring>

#ifdef USE_ZSTD
#include <zstd.h>
#endif

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
};

static std::string strip_spaces(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char ch : s) if (!std::isspace(static_cast<unsigned char>(ch))) out.push_back(ch);
    return out;
}

static int parse_signed_int(const std::string& s, std::size_t& i) {
    bool neg = false;
    if (i < s.size() && (s[i] == '+' || s[i] == '-')) { neg = (s[i] == '-'); ++i; }
    long long v = 0;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) { v = 10 * v + (s[i] - '0'); ++i; }
    if (neg) v = -v;
    return static_cast<int>(v);
}

static Row parse_line(const std::string& line, int& max_var_seen) {
    std::string s = strip_spaces(line);
    if (s.empty()) return Row{};
    auto eq = s.find('=');
    if (eq == std::string::npos) throw std::runtime_error("Missing '='");
    std::string lhs = s.substr(0, eq), rhs_s = s.substr(eq + 1);
    std::size_t rp = 0;
    int rhs = parse_signed_int(rhs_s, rp);
    std::unordered_map<int, int> merged;
    std::size_t i = 0;
    while (i < lhs.size()) {
        int sign = +1;
        while (i < lhs.size() && (lhs[i] == '+' || lhs[i] == '-')) { if (lhs[i] == '-') sign = -sign; ++i; }
        int coeff = 1;
        std::size_t c0 = i;
        while (i < lhs.size() && std::isdigit(static_cast<unsigned char>(lhs[i]))) ++i;
        if (i > c0) { coeff = std::stoi(lhs.substr(c0, i - c0)); if (i < lhs.size() && lhs[i] == '*') ++i; }
        i += 2; // skip x_
        std::size_t v0 = i;
        while (i < lhs.size() && std::isdigit(static_cast<unsigned char>(lhs[i]))) ++i;
        int var = std::stoi(lhs.substr(v0, i - v0)) - 1;
        max_var_seen = std::max(max_var_seen, var);
        merged[var] += sign * coeff;
    }
    Row row; row.rhs = rhs;
    std::vector<std::pair<int, int>> items;
    for (const auto& kv : merged) if (kv.second != 0) items.push_back(kv);
    std::sort(items.begin(), items.end());
    for (const auto& kv : items) { row.vars.push_back(kv.first); row.coeffs.push_back(kv.second); }
    return row;
}

static Model load_model(const std::string& filename) {
    std::ifstream fin(filename);
    if (!fin) throw std::runtime_error("Cannot open file: " + filename);
    Model M; std::string line; int max_v = -1; std::vector<Row> rows;
    while (std::getline(fin, line)) { if (!strip_spaces(line).empty()) rows.push_back(parse_line(line, max_v)); }
    M.nvars = max_v + 1; M.nrows = (int)rows.size();
    M.var_adj.assign(M.nvars, {});
    std::vector<int> score(M.nvars, 0), degree(M.nvars, 0);
    M.row_offsets.push_back(0);
    for (int r = 0; r < M.nrows; ++r) {
        M.rhs.push_back(rows[r].rhs);
        int mc = 0;
        for (int k = 0; k < (int)rows[r].vars.size(); ++k) {
            int v = rows[r].vars[k], c = rows[r].coeffs[k];
            M.all_vars.push_back(v); M.all_coeffs.push_back(c);
            M.var_adj[v].push_back({r, c});
            score[v] += std::abs(c); degree[v]++;
            mc = std::max(mc, std::abs(c));
        }
        M.row_max_coeff.push_back(mc);
        M.row_offsets.push_back((int)M.all_vars.size());
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
    const Model* M;
    std::vector<signed char> x;
    std::vector<int> target, pos_rem, neg_rem;
    std::vector<int> queue, row_seen, row_saved, var_trail;
    struct RowSnap { int r, t, p, n; };
    std::vector<RowSnap> row_trail;
    int stamp = 1;

    WorkerState(const Model& model) : M(&model) {
        x.assign(M->nvars, -1);
        target.assign(M->nrows, 0); pos_rem.assign(M->nrows, 0); neg_rem.assign(M->nrows, 0);
        row_seen.assign(M->nrows, 0); row_saved.assign(M->nrows, 0);
        for (int r = 0; r < M->nrows; ++r) {
            target[r] = M->rhs[r];
            int p = 0, n = 0;
            for (int i = M->row_offsets[r]; i < M->row_offsets[r+1]; ++i) {
                int c = M->all_coeffs[i]; if (c > 0) p += c; else n += c;
            }
            pos_rem[r] = p; neg_rem[r] = n;
        }
        queue.reserve(M->nrows); var_trail.reserve(M->nvars); row_trail.reserve(M->nrows);
    }

    inline bool assign_var(int v, int val) {
        if (x[v] != -1) return x[v] == val;
        x[v] = (signed char)val;
        var_trail.push_back(v);
        for (const auto& rc : M->var_adj[v]) {
            int r = rc.first, c = rc.second;
            if (row_saved[r] != stamp) { row_saved[r] = stamp; row_trail.push_back({r, target[r], pos_rem[r], neg_rem[r]}); }
            target[r] -= c * val;
            if (c > 0) pos_rem[r] -= c; else neg_rem[r] -= c;
            if (target[r] < neg_rem[r] || target[r] > pos_rem[r]) return false;
            if (row_seen[r] != stamp) { row_seen[r] = stamp; queue.push_back(r); }
        }
        return true;
    }

    bool propagate() {
        std::size_t head = 0;
        while (head < queue.size()) {
            int r = queue[head++];
            int mc = M->row_max_coeff[r];
            for (int i = M->row_offsets[r]; i < M->row_offsets[r+1]; ++i) {
                int v = M->all_vars[i]; if (x[v] != -1) continue;
                int c = M->all_coeffs[i];
                int t = target[r], p = pos_rem[r], n = neg_rem[r];
                if (t > n + mc - 1 && t < p - mc + 1) continue; 
                bool ok0 = (t >= n + (c < 0 ? -c : 0) && t <= p - (c > 0 ? c : 0));
                bool ok1 = (t - c >= n + (c < 0 ? -c : 0) && t - c <= p - (c > 0 ? c : 0));
                if (!ok0 && !ok1) return false;
                if (!ok0 || !ok1) { if (!assign_var(v, ok1 ? 1 : 0)) return false; }
            }
        }
        return true;
    }

    void undo(std::size_t v_cp, std::size_t r_cp) {
        for (std::size_t i = var_trail.size(); i-- > v_cp; ) x[var_trail[i]] = -1;
        var_trail.resize(v_cp);
        for (std::size_t i = row_trail.size(); i-- > r_cp; ) {
            const auto& s = row_trail[i];
            target[s.r] = s.t; pos_rem[s.r] = s.p; neg_rem[s.r] = s.n;
        }
        row_trail.resize(r_cp);
    }
};

struct Runtime {
    bool write_output = false;
    uint64_t report_every = 1000000;
    int nvars = 0;
    std::vector<std::ofstream> out_files;
};

struct TLSData {
    uint64_t count = 0, next = 1000000;
    std::chrono::steady_clock::time_point start;
    std::vector<uint64_t> pack;
#ifdef USE_ZSTD
    std::vector<char> z_buffer;
    std::vector<char> z_out;
    const size_t Z_BATCH = 1024; // Solutions per compressed block
#endif
};
static thread_local TLSData tls;

static void flush_zstd(int tid, Runtime& rt) {
#ifdef USE_ZSTD
    if (tls.z_buffer.empty()) return;
    size_t const cSize = ZSTD_compress(tls.z_out.data(), tls.z_out.size(), tls.z_buffer.data(), tls.z_buffer.size(), 3);
    if (ZSTD_isError(cSize)) throw std::runtime_error("ZSTD compression failed");
    uint32_t sz = (uint32_t)cSize;
    rt.out_files[tid].write((char*)&sz, 4);
    rt.out_files[tid].write(tls.z_out.data(), cSize);
    tls.z_buffer.clear();
#endif
}

static void record(const WorkerState& S, Runtime& rt) {
    tls.count++;
    if (rt.write_output) {
        int tid = omp_get_thread_num();
        std::fill(tls.pack.begin(), tls.pack.end(), 0);
        for (int v = 0; v < S.M->nvars; ++v) if (S.x[v] == 1) tls.pack[v >> 6] |= (1ULL << (v & 63));
        
#ifdef USE_ZSTD
        size_t sol_bytes = tls.pack.size() * 8;
        size_t offset = tls.z_buffer.size();
        tls.z_buffer.resize(offset + sol_bytes);
        std::memcpy(tls.z_buffer.data() + offset, tls.pack.data(), sol_bytes);
        if (tls.z_buffer.size() >= tls.Z_BATCH * sol_bytes) flush_zstd(tid, rt);
#else
        rt.out_files[tid].write((char*)tls.pack.data(), tls.pack.size() * 8);
#endif
    }
    if (tls.count >= tls.next) {
        auto d = std::chrono::duration<double>(std::chrono::steady_clock::now() - tls.start).count();
        std::cerr << "[thread " << omp_get_thread_num() << "] count=" << tls.count << " rate=" << (d > 0 ? tls.count/d : 0) << " sol/s" << std::endl;
        tls.next += rt.report_every;
    }
}

static void dfs_seq(WorkerState& S, int v_idx, Runtime& rt) {
    while (v_idx < S.M->nvars && S.x[S.M->var_order[v_idx]] != -1) v_idx++;
    if (v_idx == S.M->nvars) { record(S, rt); return; }
    int v = S.M->var_order[v_idx];
    for (int val = 0; val <= 1; ++val) {
        S.stamp++; S.queue.clear();
        std::size_t v_cp = S.var_trail.size(), r_cp = S.row_trail.size();
        if (S.assign_var(v, val) && S.propagate()) dfs_seq(S, v_idx + 1, rt);
        S.undo(v_cp, r_cp);
    }
}

static void dfs_task(WorkerState S, int v_idx, int depth, Runtime& rt) {
    while (v_idx < S.M->nvars && S.x[S.M->var_order[v_idx]] != -1) v_idx++;
    if (v_idx == S.M->nvars) { record(S, rt); return; }
    if (depth <= 0) { dfs_seq(S, v_idx, rt); return; }
    int v = S.M->var_order[v_idx];
    for (int val = 0; val <= 1; ++val) {
        WorkerState child = S; child.stamp++; child.queue.clear();
        if (child.assign_var(v, val) && child.propagate()) {
            #pragma omp task default(none) firstprivate(child, v_idx, depth) shared(rt)
            dfs_task(child, v_idx + 1, depth - 1, rt);
        }
    }
    #pragma omp taskwait
}

int main(int argc, char** argv) {
    if (argc < 2) return 0;
    int threads = std::thread::hardware_concurrency(), depth = 8;
    uint64_t report = 1000000;
    bool count_only = false; std::string out_dir;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--threads") threads = std::stoi(argv[++i]);
        else if (a == "--spawn-depth") depth = std::stoi(argv[++i]);
        else if (a == "--report-every") report = std::stoull(argv[++i]);
        else if (a == "--count-only") count_only = true;
        else if (a == "--out-dir") out_dir = argv[++i];
    }
    Model M = load_model(argv[1]);
    Runtime rt; rt.report_every = report; rt.nvars = M.nvars;
    rt.write_output = !count_only && !out_dir.empty();
    if (rt.write_output) {
        std::filesystem::create_directories(out_dir);
        rt.out_files.resize(threads);
        for (int i = 0; i < threads; ++i) {
            rt.out_files[i].open(out_dir + "/worker_" + std::to_string(i) + ".bin", std::ios::binary);
#ifdef USE_ZSTD
            rt.out_files[i].write("ZSD1", 4);
#endif
        }
    }
    std::vector<uint64_t> totals(threads, 0);
    auto start = std::chrono::steady_clock::now();
    omp_set_num_threads(threads);
    #pragma omp parallel default(none) shared(M, rt, depth, totals)
    {
        tls.start = std::chrono::steady_clock::now(); tls.next = rt.report_every; 
        tls.pack.assign((M.nvars + 63) / 64, 0);
#ifdef USE_ZSTD
        size_t sol_bytes = tls.pack.size() * 8;
        tls.z_buffer.reserve(tls.Z_BATCH * sol_bytes);
        tls.z_out.resize(ZSTD_compressBound(tls.Z_BATCH * sol_bytes));
#endif
        #pragma omp single
        {
            WorkerState root(M); root.stamp++; root.queue.clear();
            for (int r = 0; r < M.nrows; ++r) { root.row_seen[r] = root.stamp; root.queue.push_back(r); }
            if (root.propagate()) dfs_task(root, 0, depth, rt);
        }
        flush_zstd(omp_get_thread_num(), rt);
        totals[omp_get_thread_num()] = tls.count;
    }
    uint64_t total = 0; for (auto t : totals) total += t;
    auto end = std::chrono::steady_clock::now();
    double s = std::chrono::duration<double>(end - start).count();
    std::cout << "Total solutions: " << total << "\nElapsed: " << s << " s\nAverage: " << (s > 0 ? total/s : 0) << " sol/s" << std::endl;
    return 0;
}
