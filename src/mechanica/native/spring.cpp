#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
using torch::indexing::Slice;

namespace {

void check_shape(bool condition, const std::string& message) {
    TORCH_CHECK(condition, message);
}

double dtype_epsilon(const torch::Tensor& tensor) {
    if (tensor.scalar_type() == torch::kFloat64) {
        return std::numeric_limits<double>::epsilon();
    }
    return std::numeric_limits<float>::epsilon();
}

torch::Tensor expand_to_length_rank(torch::Tensor value, const torch::Tensor& length) {
    while (value.dim() < length.dim()) {
        value = value.unsqueeze(0);
    }
    return value;
}

}  // namespace

torch::Tensor skew3(torch::Tensor vector) {
    check_shape(vector.size(-1) == 3, "vector must end in dimension 3");
    auto components = vector.unbind(-1);
    auto zero = torch::zeros_like(components[0]);
    auto shape = vector.sizes().vec();
    shape.push_back(3);
    return torch::stack({
        zero, -components[2], components[1],
        components[2], zero, -components[0],
        -components[1], components[0], zero}, -1).reshape(shape);
}

torch::Tensor so3_exp_kernel(torch::Tensor rotation_vector) {
    check_shape(rotation_vector.size(-1) == 3, "rotation_vector must end in dimension 3");
    auto theta2 = rotation_vector.square().sum(-1, true);
    auto safe = theta2.clamp_min(dtype_epsilon(rotation_vector));
    auto theta = safe.sqrt();
    auto small = theta2 < 1e-8;
    auto a = torch::where(
        small, 1 - theta2 / 6 + theta2.square() / 120, theta.sin() / theta);
    auto b = torch::where(
        small, 0.5 - theta2 / 24 + theta2.square() / 720, (1 - theta.cos()) / safe);
    auto matrix = skew3(rotation_vector);
    auto identity = torch::eye(3, rotation_vector.options());
    return identity + a.unsqueeze(-1) * matrix + b.unsqueeze(-1) * matrix.matmul(matrix);
}

torch::Tensor make_transform(torch::Tensor rotation, torch::Tensor translation) {
    auto top = torch::cat({rotation, translation.unsqueeze(-1)}, -1);
    auto bottom = torch::zeros({rotation.size(0), 1, 4}, rotation.options());
    bottom.index_put_({Slice(), 0, 3}, 1);
    return torch::cat({top, bottom}, -2);
}

torch::Tensor forward_kinematics_kernel(
    torch::Tensor parents,
    torch::Tensor joint_types,
    torch::Tensor q_indices,
    torch::Tensor axes,
    torch::Tensor origins,
    torch::Tensor multipliers,
    torch::Tensor offsets,
    torch::Tensor q) {
    check_shape(q.dim() >= 1, "q must have at least one dimension");
    const auto links = parents.numel();
    const auto batches = q.numel() / q.size(-1);
    auto flat_q = q.reshape({batches, q.size(-1)});
    axes = axes.to(q.options());
    origins = origins.to(q.options());
    multipliers = multipliers.to(q.options());
    offsets = offsets.to(q.options());
    std::vector<torch::Tensor> world;
    world.reserve(links);
    for (int64_t link = 0; link < links; ++link) {
        const auto coordinate = q_indices[link].item<int64_t>();
        auto motion = torch::eye(4, q.options()).expand({batches, 4, 4});
        if (coordinate >= 0) {
            auto value = multipliers[link] * flat_q.index({Slice(), coordinate}) + offsets[link];
            auto axis = axes[link].expand({batches, 3});
            if (joint_types[link].item<int64_t>() == 2) {
                auto rotation = torch::eye(3, q.options()).expand({batches, 3, 3});
                motion = make_transform(rotation, axis * value.unsqueeze(-1));
            } else {
                motion = make_transform(so3_exp_kernel(axis * value.unsqueeze(-1)), torch::zeros_like(axis));
            }
        }
        auto local = origins[link].unsqueeze(0).matmul(motion);
        const auto parent = parents[link].item<int64_t>();
        world.push_back(parent < 0 ? local : world[parent].matmul(local));
    }
    auto shape = q.sizes().slice(0, q.dim() - 1).vec();
    shape.insert(shape.end(), {links, 4, 4});
    return torch::stack(world, 1).reshape(shape);
}

torch::Tensor hooke_spring_force(
    torch::Tensor positions,
    torch::Tensor edges,
    torch::Tensor rest_lengths,
    torch::Tensor stiffness,
    py::object velocities_obj,
    torch::Tensor damping) {
    check_shape(positions.dim() >= 2, "positions must be shaped (..., bodies, dim)");
    check_shape(edges.dim() == 2 && edges.size(-1) == 2, "edges must be shaped (springs, 2)");
    check_shape(positions.is_floating_point(), "positions must be floating point");

    auto edge_index = edges.to(positions.device(), torch::kLong);
    auto i = edge_index.index({Slice(), 0});
    auto j = edge_index.index({Slice(), 1});

    const auto bodies = positions.size(-2);
    const auto dim = positions.size(-1);
    const auto batches = positions.numel() / (bodies * dim);
    auto flat_positions = positions.reshape({batches, bodies, dim});
    auto delta = flat_positions.index({Slice(), j, Slice()}) -
                 flat_positions.index({Slice(), i, Slice()});
    auto length = (delta * delta).sum(-1).sqrt().clamp_min(dtype_epsilon(positions));
    auto direction = delta / length.unsqueeze(-1);

    auto rest = expand_to_length_rank(rest_lengths, length).expand_as(length);
    auto k = expand_to_length_rank(stiffness, length).expand_as(length);
    auto magnitude = k * (length - rest);

    if (!velocities_obj.is_none()) {
        auto velocities = velocities_obj.cast<torch::Tensor>();
        check_shape(
            velocities.sizes() == positions.sizes(),
            "velocities must have the same shape as positions");
        auto flat_velocities = velocities.reshape({batches, bodies, dim});
        auto rel_vel = flat_velocities.index({Slice(), j, Slice()}) -
                       flat_velocities.index({Slice(), i, Slice()});
        auto c = expand_to_length_rank(damping, length).expand_as(length);
        magnitude = magnitude + c * (rel_vel * direction).sum(-1);
    }

    auto force_edges = magnitude.unsqueeze(-1) * direction;
    auto offsets = torch::arange(batches, i.options()).unsqueeze(-1) * bodies;
    auto flat_i = (i.unsqueeze(0) + offsets).reshape({-1});
    auto flat_j = (j.unsqueeze(0) + offsets).reshape({-1});
    auto forces = torch::zeros_like(positions).reshape({-1, dim});
    auto flat_force = force_edges.reshape({-1, dim});
    forces.index_add_(0, flat_i, flat_force);
    forces.index_add_(0, flat_j, -flat_force);
    return forces.reshape_as(positions);
}

torch::Tensor gravity_neighbor_list(torch::Tensor positions, double cutoff) {
    check_shape(positions.device().is_cpu(), "neighbor list currently requires CPU positions");
    check_shape(positions.dim() == 2, "positions must be shaped (bodies, dim)");
    check_shape(positions.is_floating_point(), "positions must be floating point");
    TORCH_CHECK(cutoff > 0, "cutoff must be positive");

    auto values = positions.contiguous();
    const auto bodies = values.size(0);
    const auto dim = values.size(1);
    check_shape(dim > 0 && dim <= 3, "neighbor list supports one to three dimensions");
    const auto cutoff2 = cutoff * cutoff;
    struct Cell {
        int64_t x, y, z;
        bool operator==(const Cell& other) const {
            return x == other.x && y == other.y && z == other.z;
        }
    };
    struct CellHash {
        size_t operator()(const Cell& cell) const {
            size_t seed = std::hash<int64_t>{}(cell.x);
            seed ^= std::hash<int64_t>{}(cell.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= std::hash<int64_t>{}(cell.z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
        }
    };
    std::vector<std::pair<int64_t, int64_t>> pairs;
    AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "gravity_neighbor_list", [&] {
        const auto* data = values.data_ptr<scalar_t>();
        std::unordered_map<Cell, std::vector<int64_t>, CellHash> cells;
        for (int64_t j = 0; j < bodies; ++j) {
            const auto coordinate = [&](int64_t d) {
                return d < dim
                    ? static_cast<int64_t>(std::floor(data[j * dim + d] / cutoff))
                    : 0;
            };
            const Cell cell{coordinate(0), coordinate(1), coordinate(2)};
            for (int64_t dx = -1; dx <= 1; ++dx) {
                for (int64_t dy = dim > 1 ? -1 : 0; dy <= (dim > 1 ? 1 : 0); ++dy) {
                    for (int64_t dz = dim > 2 ? -1 : 0; dz <= (dim > 2 ? 1 : 0); ++dz) {
                        const Cell neighbor{cell.x + dx, cell.y + dy, cell.z + dz};
                        auto found = cells.find(neighbor);
                        if (found == cells.end()) continue;
                        for (const auto i : found->second) {
                            scalar_t distance2 = 0;
                            for (int64_t d = 0; d < dim; ++d) {
                                const auto delta = data[j * dim + d] - data[i * dim + d];
                                distance2 += delta * delta;
                            }
                            if (distance2 < cutoff2) pairs.emplace_back(i, j);
                        }
                    }
                }
            }
            cells[cell].push_back(j);
        }
    });
    std::sort(pairs.begin(), pairs.end());
    auto result = torch::empty(
        {static_cast<int64_t>(pairs.size()), 2},
        torch::TensorOptions().dtype(torch::kLong));
    auto* output = result.data_ptr<int64_t>();
    for (size_t index = 0; index < pairs.size(); ++index) {
        output[2 * index] = pairs[index].first;
        output[2 * index + 1] = pairs[index].second;
    }
    return result;
}

torch::Tensor pairwise_gravity_force(
    torch::Tensor positions,
    torch::Tensor masses,
    double gravitational_constant,
    double softening) {
    check_shape(positions.dim() == 2, "positions must be shaped (bodies, dim)");
    check_shape(positions.is_floating_point(), "positions must be floating point");

    auto mass = masses.to(positions.options());
    if (mass.dim() == 0) {
        mass = mass.expand({positions.size(0)});
    }
    check_shape(mass.dim() == 1, "masses must be a scalar or shaped (bodies,)");
    check_shape(mass.size(0) == positions.size(0), "masses length must match bodies");

    auto delta = positions.unsqueeze(0) - positions.unsqueeze(1);
    auto distance2 = (delta * delta).sum(-1) + softening * softening;
    auto eye = torch::eye(positions.size(0), positions.options().dtype(torch::kBool));
    distance2 = distance2.masked_fill(eye, 1);

    auto inv_distance3 = distance2.rsqrt() / distance2;
    inv_distance3 = inv_distance3.masked_fill(eye, 0);

    auto mass_product = mass.unsqueeze(0) * mass.unsqueeze(1);
    auto magnitude = gravitational_constant * mass_product * inv_distance3;
    return (magnitude.unsqueeze(-1) * delta).sum(1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("so3_exp", &so3_exp_kernel, "Batched SO(3) exponential map");
    m.def("forward_kinematics", &forward_kinematics_kernel, "Batched tree kinematics");
    m.def(
        "hooke_spring_force",
        &hooke_spring_force,
        "Hooke spring forces for particle graphs");
    m.def(
        "pairwise_gravity_force",
        &pairwise_gravity_force,
        "Pairwise gravitational forces for particle systems");
    m.def("gravity_neighbor_list", &gravity_neighbor_list, "Cutoff neighbor pairs");
}

TORCH_LIBRARY(mechanica, m) {
    m.def("so3_exp(Tensor rotation_vector) -> Tensor");
    m.def("forward_kinematics(Tensor parents, Tensor joint_types, Tensor q_indices, Tensor axes, Tensor origins, Tensor multipliers, Tensor offsets, Tensor q) -> Tensor");
}

TORCH_LIBRARY_IMPL(mechanica, CompositeImplicitAutograd, m) {
    m.impl("so3_exp", &so3_exp_kernel);
    m.impl("forward_kinematics", &forward_kinematics_kernel);
}
