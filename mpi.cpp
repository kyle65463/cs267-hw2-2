#include "common.h"
#include <algorithm>
#include <cmath>
#include <mpi.h>
#include <vector>

#define MAX_PER_CELL 8

// Static global variables
static double grid_cell_size;
static int grid_nx, grid_ny;
static int start_row, end_row, local_ny;
static int lower_neighbor, upper_neighbor;
static std::vector<std::vector<particle_t>> ghost_lower;
static std::vector<std::vector<particle_t>> ghost_upper;
static std::vector<particle_t> local_parts;
static std::vector<int> all_start_rows;
static std::vector<int> all_end_rows;
static int step_counter = 0;

struct cell_t {
    int particles[MAX_PER_CELL];
    int size;
};

static std::vector<cell_t> local_cells;

inline int local_cell_index(int cx, int local_cy) {
    return local_cy * grid_nx + cx;
}

inline void apply_force(particle_t& p1, particle_t& p2) {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);
    double coef = (1.0 - cutoff / r) / r2 / mass;
    double fx = coef * dx;
    double fy = coef * dy;
    p1.ax += fx;
    p1.ay += fy;
    p2.ax -= fx;
    p2.ay -= fy;
}

inline void apply_force_local(particle_t& local_p, const particle_t& other_p) {
    double dx = other_p.x - local_p.x;
    double dy = other_p.y - local_p.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);
    double coef = (1.0 - cutoff / r) / r2 / mass;
    double fx = coef * dx;
    double fy = coef * dy;
    local_p.ax += fx;
    local_p.ay += fy;
}

inline void move(particle_t& p, double size) {
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }
    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Compute grid parameters
    grid_cell_size = cutoff * 2.5;
    grid_nx = static_cast<int>(size / grid_cell_size) + 1;
    grid_ny = static_cast<int>(size / grid_cell_size) + 1;

    // Assign rows to processors
    int rows_per_proc = grid_ny / num_procs;
    int extra = grid_ny % num_procs;
    if (rank < extra) {
        start_row = rank * (rows_per_proc + 1);
        local_ny = rows_per_proc + 1;
    } else {
        start_row = rank * rows_per_proc + extra;
        local_ny = rows_per_proc;
    }
    end_row = start_row + local_ny - 1;

    // Set neighbor ranks
    lower_neighbor = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    upper_neighbor = (rank < num_procs - 1) ? rank + 1 : MPI_PROC_NULL;

    // Filter local particles
    local_parts.clear();
    for (int i = 0; i < num_parts; i++) {
        int cy = static_cast<int>(parts[i].y / grid_cell_size);
        if (cy >= start_row && cy <= end_row) {
            local_parts.push_back(parts[i]);
        }
    }

    // Initialize local cells
    local_cells.resize(std::max(local_ny, 0) * grid_nx);
    for (auto& cell : local_cells) {
        cell.size = 0;
    }

    // Initialize ghost particle arrays
    ghost_lower.resize(grid_nx);
    ghost_upper.resize(grid_nx);

    // Share row assignments
    all_start_rows.resize(num_procs);
    all_end_rows.resize(num_procs);
    MPI_Allgather(&start_row, 1, MPI_INT,
                  all_start_rows.data(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&end_row, 1, MPI_INT,
                  all_end_rows.data(), 1, MPI_INT, MPI_COMM_WORLD);
}

static void gather_row_particles(
    int global_row,
    std::vector<int>& counts,
    std::vector<particle_t>& buf
) {
    // If local_ny == 0, or global_row not in [start_row,end_row], gather zero
    if (local_ny == 0 || global_row < start_row || global_row > end_row) {
        counts.assign(grid_nx, 0);
        buf.clear();
        return;
    }
    int local_row = global_row - start_row;
    counts.resize(grid_nx, 0);

    // Count how many in each cell of that row
    int total = 0;
    for (int cx = 0; cx < grid_nx; cx++) {
        int cidx = local_cell_index(cx, local_row);
        counts[cx] = local_cells[cidx].size;
        total += local_cells[cidx].size;
    }
    buf.resize(total);

    // Copy them out
    int offset = 0;
    for (int cx = 0; cx < grid_nx; cx++) {
        int cidx = local_cell_index(cx, local_row);
        for (int j = 0; j < local_cells[cidx].size; j++) {
            buf[offset++] = local_parts[local_cells[cidx].particles[j]];
        }
    }
}

static void exchange_ghost_rows(double size, int rank, int num_procs) {
    // Clear any leftover from previous step
    for (auto &v : ghost_lower) {
        v.clear();
    }
    for (auto &v : ghost_upper) {
        v.clear();
    }

    // Two-phase approach with nonblocking calls:
    //   Phase A: exchange row "counts"
    //   Phase B: exchange row "data"

    std::vector<MPI_Request> requests;

    // Gather row data
    std::vector<int> upper_send_counts(grid_nx, 0), upper_recv_counts(grid_nx, 0);
    std::vector<particle_t> upper_send_buf, upper_recv_buf;
    gather_row_particles(end_row, upper_send_counts, upper_send_buf);

    std::vector<int> lower_send_counts(grid_nx, 0), lower_recv_counts(grid_nx, 0);
    std::vector<particle_t> lower_send_buf, lower_recv_buf;
    gather_row_particles(start_row, lower_send_counts, lower_send_buf);

    // PHASE A: exchange counts
    MPI_Request req;

    // Send/receive counts with upper neighbor
    if (upper_neighbor != MPI_PROC_NULL) {
        MPI_Irecv(upper_recv_counts.data(), grid_nx, MPI_INT,
                  upper_neighbor, 2, MPI_COMM_WORLD, &req);
        requests.push_back(req);
        MPI_Isend(upper_send_counts.data(), grid_nx, MPI_INT,
                  upper_neighbor, 0, MPI_COMM_WORLD, &req);
        requests.push_back(req);
    }

    // Send/receive counts with lower neighbor
    if (lower_neighbor != MPI_PROC_NULL) {
        MPI_Irecv(lower_recv_counts.data(), grid_nx, MPI_INT,
                  lower_neighbor, 0, MPI_COMM_WORLD, &req);
        requests.push_back(req);
        MPI_Isend(lower_send_counts.data(), grid_nx, MPI_INT,
                  lower_neighbor, 2, MPI_COMM_WORLD, &req);
        requests.push_back(req);
    }

    // Wait for all count exchanges
    if (!requests.empty()) {
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }
    requests.clear();

    // Prepare receive buffers
    int total_upper_recv = 0;
    for (int cx = 0; cx < grid_nx; cx++) {
        total_upper_recv += upper_recv_counts[cx];
    }
    upper_recv_buf.resize(total_upper_recv);

    int total_lower_recv = 0;
    for (int cx = 0; cx < grid_nx; cx++) {
        total_lower_recv += lower_recv_counts[cx];
    }
    lower_recv_buf.resize(total_lower_recv);

    // PHASE B: exchange data
    // With upper neighbor
    if (upper_neighbor != MPI_PROC_NULL) {
        MPI_Irecv(upper_recv_buf.data(), total_upper_recv, PARTICLE,
                  upper_neighbor, 3, MPI_COMM_WORLD, &req);
        requests.push_back(req);

        MPI_Isend(upper_send_buf.data(), (int)upper_send_buf.size(), PARTICLE,
                  upper_neighbor, 1, MPI_COMM_WORLD, &req);
        requests.push_back(req);
    }

    // With lower neighbor
    if (lower_neighbor != MPI_PROC_NULL) {
        MPI_Irecv(lower_recv_buf.data(), total_lower_recv, PARTICLE,
                  lower_neighbor, 1, MPI_COMM_WORLD, &req);
        requests.push_back(req);

        MPI_Isend(lower_send_buf.data(), (int)lower_send_buf.size(), PARTICLE,
                  lower_neighbor, 3, MPI_COMM_WORLD, &req);
        requests.push_back(req);
    }

    // Wait for data-phase requests
    if (!requests.empty()) {
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }

    // Store received ghost data
    {
        int offset = 0;
        for (int cx = 0; cx < grid_nx; cx++) {
            int ccount = upper_recv_counts[cx];
            ghost_upper[cx].resize(ccount);
            for (int i = 0; i < ccount; i++) {
                ghost_upper[cx][i] = upper_recv_buf[offset + i];
            }
            offset += ccount;
        }
    }
    {
        int offset = 0;
        for (int cx = 0; cx < grid_nx; cx++) {
            int ccount = lower_recv_counts[cx];
            ghost_lower[cx].resize(ccount);
            for (int i = 0; i < ccount; i++) {
                ghost_lower[cx][i] = lower_recv_buf[offset + i];
            }
            offset += ccount;
        }
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    step_counter++;

    // Clear local cells
    for (auto& cell : local_cells) {
        cell.size = 0;
    }

    // Reassign local particles to cells
    // (If local_ny=0, local_cells is empty, so this loop is effectively safe)
    for (size_t i = 0; i < local_parts.size(); i++) {
        const auto& p = local_parts[i];
        int cx = static_cast<int>(p.x / grid_cell_size);
        int cy = static_cast<int>(p.y / grid_cell_size);
        int local_cy = cy - start_row;
        if (local_cy >= 0 && local_cy < local_ny &&
            cx >= 0 && cx < grid_nx) {
            int cidx = local_cell_index(cx, local_cy);
            if (local_cells[cidx].size < MAX_PER_CELL) {
                local_cells[cidx].particles[local_cells[cidx].size] = i;
                local_cells[cidx].size++;
            }
        }
    }

    // Exchange ghost rows with upper/lower neighbor (non-blocking, but always called)
    exchange_ghost_rows(size, rank, num_procs);

    // Reset accelerations
    for (auto& p : local_parts) {
        p.ax = 0.0;
        p.ay = 0.0;
    }

    // Apply forces (local cell vs local cell, including neighboring cells)
    // If local_ny=0, this loop won't do anything.
    for (int local_cy = 0; local_cy < local_ny; local_cy++) {
        int global_cy = start_row + local_cy;
        for (int cx = 0; cx < grid_nx; cx++) {
            int local_i = local_cell_index(cx, local_cy);
            cell_t& c1 = local_cells[local_i];

            // We look at the 3x3 block of neighbors around (cx, global_cy)
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = cx + dx;
                    int ny = global_cy + dy;
                    if (nx < 0 || nx >= grid_nx || ny < 0 || ny >= grid_ny)
                        continue;

                    // if inside my local domain
                    if (ny >= start_row && ny <= end_row) {
                        // The neighbor cell is local
                        int local_j = local_cell_index(nx, ny - start_row);
                        cell_t& c2 = local_cells[local_j];
                        if (local_j > local_i) {
                            for (int idx1 = 0; idx1 < c1.size; idx1++) {
                                for (int idx2 = 0; idx2 < c2.size; idx2++) {
                                    apply_force(local_parts[c1.particles[idx1]],
                                                local_parts[c2.particles[idx2]]);
                                }
                            }
                        } else if (local_j == local_i) {
                            // same cell: pairwise
                            for (int idx1 = 0; idx1 < c1.size; idx1++) {
                                for (int idx2 = idx1 + 1; idx2 < c1.size; idx2++) {
                                    apply_force(local_parts[c1.particles[idx1]],
                                                local_parts[c1.particles[idx2]]);
                                }
                            }
                        }
                    }
                    else {
                        // The neighbor cell is in a ghost row
                        if (ny == end_row + 1 && upper_neighbor != MPI_PROC_NULL) {
                            const auto& ghost_c2 = ghost_upper[nx];
                            for (int idx1 = 0; idx1 < c1.size; idx1++) {
                                for (const auto& p2 : ghost_c2) {
                                    apply_force_local(local_parts[c1.particles[idx1]], p2);
                                }
                            }
                        } else if (ny == start_row - 1 && lower_neighbor != MPI_PROC_NULL) {
                            const auto& ghost_c2 = ghost_lower[nx];
                            for (int idx1 = 0; idx1 < c1.size; idx1++) {
                                for (const auto& p2 : ghost_c2) {
                                    apply_force_local(local_parts[c1.particles[idx1]], p2);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Move particles
    for (auto& p : local_parts) {
        move(p, size);
    }

    // Migrate particles (Alltoallv) so that each rank only owns the rows assigned to it
    std::vector<int> send_counts(num_procs, 0);
    for (const auto& p : local_parts) {
        int cy = static_cast<int>(p.y / grid_cell_size);
        if (cy < 0) cy = 0;
        if (cy >= grid_ny) cy = grid_ny - 1;
        int target_rank = -1;
        // Find which rank owns row 'cy'
        for (int r = 0; r < num_procs; r++) {
            if (cy >= all_start_rows[r] && cy <= all_end_rows[r]) {
                target_rank = r;
                break;
            }
        }
        if (target_rank != rank) {
            send_counts[target_rank]++;
        }
    }

    std::vector<std::vector<particle_t>> send_lists(num_procs);
    for (const auto& p : local_parts) {
        int cy = static_cast<int>(p.y / grid_cell_size);
        if (cy < 0) cy = 0;
        if (cy >= grid_ny) cy = grid_ny - 1;
        int target_rank = -1;
        for (int r = 0; r < num_procs; r++) {
            if (cy >= all_start_rows[r] && cy <= all_end_rows[r]) {
                target_rank = r;
                break;
            }
        }
        if (target_rank != rank) {
            send_lists[target_rank].push_back(p);
        }
    }

    int total_send = 0;
    for (int r = 0; r < num_procs; r++) {
        total_send += (int)send_lists[r].size();
    }
    particle_t* send_buf = new particle_t[total_send];
    std::vector<int> send_displs(num_procs), recv_counts(num_procs), recv_displs(num_procs);

    int offset = 0;
    for (int r = 0; r < num_procs; r++) {
        send_displs[r] = offset;
        for (const auto& p : send_lists[r]) {
            send_buf[offset++] = p;
        }
    }

    // Exchange counts
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Sum up how many we get
    int total_recv = 0;
    for (int r = 0; r < num_procs; r++) {
        total_recv += recv_counts[r];
    }

    // Build displacement arrays
    offset = 0;
    for (int r = 0; r < num_procs; r++) {
        recv_displs[r] = offset;
        offset += recv_counts[r];
    }

    // Make space for incoming
    particle_t* recv_buf = new particle_t[total_recv];

    // Exchange data
    MPI_Alltoallv(send_buf, send_counts.data(), send_displs.data(), PARTICLE,
                  recv_buf, recv_counts.data(), recv_displs.data(), PARTICLE,
                  MPI_COMM_WORLD);

    // Rebuild local_parts
    std::vector<particle_t> new_local_parts;
    new_local_parts.reserve(local_parts.size() - total_send + total_recv);

    // Keep only those that still belong to me
    for (const auto& p : local_parts) {
        int cy = static_cast<int>(p.y / grid_cell_size);
        if (cy >= start_row && cy <= end_row) {
            new_local_parts.push_back(p);
        }
    }
    // Add newly received ones
    for (int i = 0; i < total_recv; i++) {
        new_local_parts.push_back(recv_buf[i]);
    }
    local_parts = std::move(new_local_parts);

    delete[] send_buf;
    delete[] recv_buf;
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    if (rank == 0) {
        // Gather counts from every rank
        std::vector<int> recv_counts(num_procs);
        recv_counts[0] = (int)local_parts.size();
        for (int r = 1; r < num_procs; r++) {
            MPI_Recv(&recv_counts[r], 1, MPI_INT, r, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        std::vector<int> recv_displs(num_procs);
        int total_recv = 0;
        for (int r = 0; r < num_procs; r++) {
            recv_displs[r] = total_recv;
            total_recv += recv_counts[r];
        }
        particle_t* all_parts = new particle_t[total_recv];

        // Copy my local parts first
        for (size_t i = 0; i < local_parts.size(); i++) {
            all_parts[i] = local_parts[i];
        }

        // Receive the data from others
        for (int r = 1; r < num_procs; r++) {
            MPI_Recv(&all_parts[recv_displs[r]], recv_counts[r], PARTICLE,
                     r, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Sort by ID so they match original ordering
        std::sort(all_parts, all_parts + total_recv,
                  [](const particle_t& a, const particle_t& b) {
                      return a.id < b.id;
                  });

        // Copy into global array
        for (int i = 0; i < num_parts; i++) {
            parts[i] = all_parts[i];
        }
        delete[] all_parts;
    } else {
        int send_count = (int)local_parts.size();
        MPI_Send(&send_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(local_parts.data(), send_count, PARTICLE, 0, 1, MPI_COMM_WORLD);
    }
}
