#include <torch/extension.h>

#include <limits>
#include <string>

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

torch::Tensor hooke_spring_force(
    torch::Tensor positions,
    torch::Tensor edges,
    torch::Tensor rest_lengths,
    torch::Tensor stiffness,
    py::object velocities_obj,
    torch::Tensor damping) {
    check_shape(positions.dim() == 2, "positions must be shaped (bodies, dim)");
    check_shape(edges.dim() == 2 && edges.size(-1) == 2, "edges must be shaped (springs, 2)");
    check_shape(positions.is_floating_point(), "positions must be floating point");

    auto i = edges.index({Slice(), 0}).to(torch::kLong);
    auto j = edges.index({Slice(), 1}).to(torch::kLong);

    auto delta = positions.index({j}) - positions.index({i});
    auto length = (delta * delta).sum(-1).sqrt().clamp_min(dtype_epsilon(positions));
    auto direction = delta / length.unsqueeze(-1);

    auto rest = expand_to_length_rank(rest_lengths, length);
    auto k = expand_to_length_rank(stiffness, length);
    auto magnitude = k * (length - rest);

    if (!velocities_obj.is_none()) {
        auto velocities = velocities_obj.cast<torch::Tensor>();
        check_shape(
            velocities.sizes() == positions.sizes(),
            "velocities must have the same shape as positions");
        auto rel_vel = velocities.index({j}) - velocities.index({i});
        auto c = expand_to_length_rank(damping, length);
        magnitude = magnitude + c * (rel_vel * direction).sum(-1);
    }

    auto force_edges = magnitude.unsqueeze(-1) * direction;
    auto forces = torch::zeros_like(positions);
    forces.index_add_(0, i, force_edges);
    forces.index_add_(0, j, -force_edges);
    return forces;
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
    m.def(
        "hooke_spring_force",
        &hooke_spring_force,
        "Hooke spring forces for particle graphs");
    m.def(
        "pairwise_gravity_force",
        &pairwise_gravity_force,
        "Pairwise gravitational forces for particle systems");
}
