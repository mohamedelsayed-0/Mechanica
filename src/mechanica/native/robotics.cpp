#include <torch/extension.h>

#include <limits>
#include <unordered_set>
#include <vector>

using torch::indexing::Slice;

namespace {

using Tensor = torch::Tensor;

void check(bool condition, const char* message) {
    TORCH_CHECK(condition, message);
}

Tensor matvec(const Tensor& matrix, const Tensor& vector) {
    return matrix.matmul(vector.unsqueeze(-1)).squeeze(-1);
}

Tensor batch_dot(const Tensor& left, const Tensor& right) {
    return (left * right).sum(-1);
}

Tensor skew(const Tensor& vector) {
    auto values = vector.unbind(-1);
    auto zero = torch::zeros_like(values[0]);
    auto shape = vector.sizes().vec();
    shape.push_back(3);
    return torch::stack({
        zero, -values[2], values[1], values[2], zero, -values[0],
        -values[1], values[0], zero}, -1).reshape(shape);
}

Tensor motion_cross_vector(const Tensor& left, const Tensor& right) {
    auto angular = left.index({Slice(), Slice(0, 3)});
    auto linear = left.index({Slice(), Slice(3, 6)});
    auto right_angular = right.index({Slice(), Slice(0, 3)});
    auto right_linear = right.index({Slice(), Slice(3, 6)});
    return torch::cat({
        torch::cross(angular, right_angular, -1),
        torch::cross(linear, right_angular, -1) +
            torch::cross(angular, right_linear, -1)}, -1);
}

Tensor force_cross_matrix(const Tensor& motion) {
    auto angular = skew(motion.index({Slice(), Slice(0, 3)}));
    auto linear = skew(motion.index({Slice(), Slice(3, 6)}));
    auto zero = torch::zeros_like(angular);
    auto cross = torch::cat({
        torch::cat({angular, zero}, -1),
        torch::cat({linear, angular}, -1)}, -2);
    return -cross.transpose(-1, -2);
}

Tensor so3_exp(const Tensor& vector) {
    auto theta2 = vector.square().sum(-1, true);
    auto safe = theta2.clamp_min(std::numeric_limits<double>::epsilon());
    auto theta = safe.sqrt();
    auto small = theta2 < 1e-8;
    auto a = torch::where(
        small, 1 - theta2 / 6 + theta2.square() / 120, theta.sin() / theta);
    auto b = torch::where(
        small, 0.5 - theta2 / 24 + theta2.square() / 720, (1 - theta.cos()) / safe);
    auto matrix = skew(vector);
    return torch::eye(3, vector.options()) + a.unsqueeze(-1) * matrix +
        b.unsqueeze(-1) * matrix.matmul(matrix);
}

Tensor make_transform(const Tensor& rotation, const Tensor& translation) {
    auto top = torch::cat({rotation, translation.unsqueeze(-1)}, -1);
    auto bottom = torch::zeros({rotation.size(0), 1, 4}, rotation.options());
    bottom.index_put_({Slice(), 0, 3}, 1);
    return torch::cat({top, bottom}, -2);
}

Tensor motion_transform(const Tensor& transform) {
    auto rotation = transform.index({Slice(), Slice(0, 3), Slice(0, 3)}).transpose(-1, -2);
    auto translation = transform.index({Slice(), Slice(0, 3), 3});
    auto zero = torch::zeros_like(rotation);
    return torch::cat({
        torch::cat({rotation, zero}, -1),
        torch::cat({-rotation.matmul(skew(translation)), rotation}, -1)}, -2);
}

struct Topology {
    std::vector<int64_t> parents;
    std::vector<int64_t> types;
    std::vector<int64_t> coordinates;
};

Topology topology(const Tensor& parents, const Tensor& types, const Tensor& coordinates) {
    auto parent_values = parents.to(torch::kCPU).contiguous();
    auto type_values = types.to(torch::kCPU).contiguous();
    auto coordinate_values = coordinates.to(torch::kCPU).contiguous();
    return {
        std::vector<int64_t>(parent_values.data_ptr<int64_t>(),
                             parent_values.data_ptr<int64_t>() + parent_values.numel()),
        std::vector<int64_t>(type_values.data_ptr<int64_t>(),
                             type_values.data_ptr<int64_t>() + type_values.numel()),
        std::vector<int64_t>(coordinate_values.data_ptr<int64_t>(),
                             coordinate_values.data_ptr<int64_t>() + coordinate_values.numel())};
}

struct TreeTerms {
    std::vector<Tensor> transforms;
    std::vector<Tensor> subspaces;
};

TreeTerms tree_terms(
    const Topology& tree,
    Tensor axes,
    Tensor origins,
    Tensor multipliers,
    Tensor offsets,
    const Tensor& q) {
    const auto batches = q.size(0);
    axes = axes.to(q.options());
    origins = origins.to(q.options());
    multipliers = multipliers.to(q.options());
    offsets = offsets.to(q.options());
    TreeTerms result;
    result.transforms.reserve(tree.parents.size());
    result.subspaces.reserve(tree.parents.size());
    for (size_t link = 0; link < tree.parents.size(); ++link) {
        auto axis = axes[link];
        auto motion = torch::eye(4, q.options()).expand({batches, 4, 4});
        if (tree.coordinates[link] >= 0) {
            auto value = multipliers[link] * q.index({Slice(), tree.coordinates[link]}) + offsets[link];
            auto expanded_axis = axis.expand({batches, 3});
            if (tree.types[link] == 2) {
                motion = make_transform(
                    torch::eye(3, q.options()).expand({batches, 3, 3}),
                    expanded_axis * value.unsqueeze(-1));
            } else {
                motion = make_transform(
                    so3_exp(expanded_axis * value.unsqueeze(-1)),
                    torch::zeros_like(expanded_axis));
            }
        }
        auto local = origins[link].unsqueeze(0).matmul(motion);
        result.transforms.push_back(motion_transform(local));
        auto scaled_axis = axis * multipliers[link];
        auto zero = torch::zeros_like(scaled_axis);
        result.subspaces.push_back(
            tree.types[link] == 2 ? torch::cat({zero, scaled_axis})
                                  : torch::cat({scaled_axis, zero}));
    }
    return result;
}

Tensor spatial_inertias(Tensor masses, Tensor centers, Tensor inertias, const Tensor& like) {
    masses = masses.to(like.options());
    centers = centers.to(like.options());
    inertias = inertias.to(like.options());
    auto cross = skew(centers);
    auto mass = masses.reshape({-1, 1, 1});
    auto upper_left = inertias + mass * cross.matmul(cross.transpose(-1, -2));
    auto upper_right = mass * cross;
    auto lower_right = mass * torch::eye(3, like.options());
    return torch::cat({
        torch::cat({upper_left, upper_right}, -1),
        torch::cat({upper_right.transpose(-1, -2), lower_right}, -1)}, -2);
}

std::vector<int64_t> sample_shape(const Tensor& q, bool matrix) {
    auto shape = q.sizes().vec();
    if (matrix) shape.push_back(q.size(-1));
    return shape;
}

Tensor rnea_impl(
    const Topology& tree,
    Tensor axes,
    Tensor origins,
    Tensor multipliers,
    Tensor offsets,
    Tensor masses,
    Tensor centers,
    Tensor inertias,
    const Tensor& q,
    const Tensor& qdot,
    const Tensor& qddot,
    Tensor gravity,
    const c10::optional<Tensor>& external_forces) {
    const auto batches = q.size(0);
    const auto dof = q.size(1);
    auto terms = tree_terms(tree, axes, origins, multipliers, offsets, q);
    auto link_inertias = spatial_inertias(masses, centers, inertias, q);
    gravity = gravity.to(q.options()).expand({batches, 3});
    auto base_acceleration = torch::cat({torch::zeros_like(gravity), -gravity}, -1);
    std::vector<Tensor> velocities, accelerations, forces;
    velocities.reserve(tree.parents.size());
    accelerations.reserve(tree.parents.size());
    forces.reserve(tree.parents.size());
    for (size_t link = 0; link < tree.parents.size(); ++link) {
        const auto coordinate = tree.coordinates[link];
        auto speed = coordinate < 0 ? torch::zeros({batches}, q.options())
                                    : qdot.index({Slice(), coordinate});
        auto acceleration_value = coordinate < 0 ? torch::zeros_like(speed)
                                                  : qddot.index({Slice(), coordinate});
        auto joint_velocity = terms.subspaces[link].unsqueeze(0) * speed.unsqueeze(-1);
        auto velocity = joint_velocity;
        auto acceleration = terms.subspaces[link].unsqueeze(0) * acceleration_value.unsqueeze(-1);
        const auto parent = tree.parents[link];
        if (parent < 0) {
            acceleration = acceleration + base_acceleration;
        } else {
            velocity = matvec(terms.transforms[link], velocities[parent]) + joint_velocity;
            acceleration = matvec(terms.transforms[link], accelerations[parent]) + acceleration +
                motion_cross_vector(velocity, joint_velocity);
        }
        auto inertia = link_inertias[link].unsqueeze(0).expand({batches, 6, 6});
        auto momentum = matvec(inertia, velocity);
        auto force = matvec(inertia, acceleration) + matvec(force_cross_matrix(velocity), momentum);
        if (external_forces.has_value()) {
            force = force - external_forces.value().to(q.options()).index({Slice(), static_cast<int64_t>(link)});
        }
        velocities.push_back(velocity);
        accelerations.push_back(acceleration);
        forces.push_back(force);
    }
    std::vector<Tensor> generalized(dof, torch::zeros({batches}, q.options()));
    for (int64_t link = tree.parents.size() - 1; link >= 0; --link) {
        const auto coordinate = tree.coordinates[link];
        if (coordinate >= 0) {
            generalized[coordinate] = generalized[coordinate] +
                batch_dot(terms.subspaces[link].unsqueeze(0), forces[link]);
        }
        const auto parent = tree.parents[link];
        if (parent >= 0) {
            forces[parent] = forces[parent] +
                matvec(terms.transforms[link].transpose(-1, -2), forces[link]);
        }
    }
    return torch::stack(generalized, -1);
}

Tensor crba_impl(
    const Topology& tree,
    Tensor axes,
    Tensor origins,
    Tensor multipliers,
    Tensor offsets,
    Tensor masses,
    Tensor centers,
    Tensor inertias,
    const Tensor& q) {
    const auto batches = q.size(0);
    const auto dof = q.size(1);
    auto terms = tree_terms(tree, axes, origins, multipliers, offsets, q);
    auto link_inertias = spatial_inertias(masses, centers, inertias, q);
    std::vector<Tensor> composite;
    composite.reserve(tree.parents.size());
    for (size_t link = 0; link < tree.parents.size(); ++link) {
        composite.push_back(link_inertias[link].unsqueeze(0).expand({batches, 6, 6}));
    }
    auto matrix = torch::zeros({batches, dof, dof}, q.options());
    for (int64_t link = tree.parents.size() - 1; link >= 0; --link) {
        const auto coordinate = tree.coordinates[link];
        if (coordinate >= 0) {
            auto force = matvec(composite[link], terms.subspaces[link].unsqueeze(0).expand({batches, 6}));
            auto diagonal = matrix.index({Slice(), coordinate, coordinate}) +
                batch_dot(terms.subspaces[link].unsqueeze(0), force);
            matrix.index_put_({Slice(), coordinate, coordinate}, diagonal);
            auto child = link;
            auto ancestor = tree.parents[child];
            while (ancestor >= 0) {
                force = matvec(terms.transforms[child].transpose(-1, -2), force);
                const auto other = tree.coordinates[ancestor];
                if (other >= 0) {
                    auto value = batch_dot(terms.subspaces[ancestor].unsqueeze(0), force);
                    if (other == coordinate) {
                        matrix.index_put_({Slice(), coordinate, coordinate},
                            matrix.index({Slice(), coordinate, coordinate}) + 2 * value);
                    } else {
                        matrix.index_put_({Slice(), coordinate, other},
                            matrix.index({Slice(), coordinate, other}) + value);
                        matrix.index_put_({Slice(), other, coordinate},
                            matrix.index({Slice(), other, coordinate}) + value);
                    }
                }
                child = ancestor;
                ancestor = tree.parents[ancestor];
            }
        }
        const auto parent = tree.parents[link];
        if (parent >= 0) {
            composite[parent] = composite[parent] + terms.transforms[link].transpose(-1, -2)
                .matmul(composite[link]).matmul(terms.transforms[link]);
        }
    }
    return matrix;
}

}  // namespace

Tensor native_rnea(
    Tensor parents, Tensor joint_types, Tensor q_indices, Tensor axes, Tensor origins,
    Tensor multipliers, Tensor offsets, Tensor masses, Tensor centers, Tensor inertias,
    Tensor q, Tensor qdot, Tensor qddot, Tensor gravity, c10::optional<Tensor> external_forces) {
    check(q.sizes() == qdot.sizes() && q.sizes() == qddot.sizes(), "state shapes must match");
    auto shape = q.sizes().vec();
    auto flat_q = q.reshape({-1, q.size(-1)});
    auto result = rnea_impl(
        topology(parents, joint_types, q_indices), axes, origins, multipliers, offsets,
        masses, centers, inertias, flat_q, qdot.reshape_as(flat_q), qddot.reshape_as(flat_q),
        gravity, external_forces.has_value()
            ? c10::optional<Tensor>(external_forces.value().reshape({-1, parents.numel(), 6}))
            : c10::nullopt);
    return result.reshape(shape);
}

Tensor native_crba(
    Tensor parents, Tensor joint_types, Tensor q_indices, Tensor axes, Tensor origins,
    Tensor multipliers, Tensor offsets, Tensor masses, Tensor centers, Tensor inertias, Tensor q) {
    auto result = crba_impl(
        topology(parents, joint_types, q_indices), axes, origins, multipliers, offsets,
        masses, centers, inertias, q.reshape({-1, q.size(-1)}));
    return result.reshape(sample_shape(q, true));
}

Tensor native_aba(
    Tensor parents, Tensor joint_types, Tensor q_indices, Tensor axes, Tensor origins,
    Tensor multipliers, Tensor offsets, Tensor masses, Tensor centers, Tensor inertias,
    Tensor q, Tensor qdot, Tensor generalized_forces, Tensor gravity) {
    check(q.sizes() == qdot.sizes() && q.sizes() == generalized_forces.sizes(),
          "state and force shapes must match");
    auto tree = topology(parents, joint_types, q_indices);
    auto shape = q.sizes().vec();
    auto flat_q = q.reshape({-1, q.size(-1)});
    auto flat_qdot = qdot.reshape_as(flat_q);
    auto flat_forces = generalized_forces.reshape_as(flat_q);
    std::unordered_set<int64_t> active;
    bool coupled = false;
    for (const auto coordinate : tree.coordinates) {
        if (coordinate >= 0 && !active.insert(coordinate).second) coupled = true;
    }
    if (coupled) {
        auto zero = torch::zeros_like(flat_q);
        auto bias = rnea_impl(tree, axes, origins, multipliers, offsets, masses, centers,
                              inertias, flat_q, flat_qdot, zero, gravity, c10::nullopt);
        auto mass = crba_impl(tree, axes, origins, multipliers, offsets, masses, centers,
                              inertias, flat_q);
        return torch::linalg_solve(mass, (flat_forces - bias).unsqueeze(-1)).squeeze(-1).reshape(shape);
    }

    const auto batches = flat_q.size(0);
    auto terms = tree_terms(tree, axes, origins, multipliers, offsets, flat_q);
    auto link_inertias = spatial_inertias(masses, centers, inertias, flat_q);
    std::vector<Tensor> articulated, velocities, bias_accelerations, bias_forces;
    for (size_t link = 0; link < tree.parents.size(); ++link) {
        articulated.push_back(link_inertias[link].unsqueeze(0).expand({batches, 6, 6}));
        const auto coordinate = tree.coordinates[link];
        auto speed = coordinate < 0 ? torch::zeros({batches}, flat_q.options())
                                    : flat_qdot.index({Slice(), coordinate});
        auto joint_velocity = terms.subspaces[link].unsqueeze(0) * speed.unsqueeze(-1);
        const auto parent = tree.parents[link];
        auto velocity = parent < 0 ? joint_velocity
                                   : matvec(terms.transforms[link], velocities[parent]) + joint_velocity;
        velocities.push_back(velocity);
        bias_accelerations.push_back(motion_cross_vector(velocity, joint_velocity));
        bias_forces.push_back(matvec(
            force_cross_matrix(velocity), matvec(articulated[link], velocity)));
    }

    std::vector<Tensor> capital_u(tree.parents.size()), d(tree.parents.size()), u(tree.parents.size());
    for (int64_t link = tree.parents.size() - 1; link >= 0; --link) {
        const auto coordinate = tree.coordinates[link];
        auto reduced_inertia = articulated[link];
        auto reduced_bias = bias_forces[link];
        if (coordinate >= 0) {
            auto subspace = terms.subspaces[link].unsqueeze(0).expand({batches, 6});
            capital_u[link] = matvec(articulated[link], subspace);
            d[link] = batch_dot(subspace, capital_u[link]);
            u[link] = flat_forces.index({Slice(), coordinate}) -
                batch_dot(subspace, bias_forces[link]);
            reduced_inertia = articulated[link] - capital_u[link].unsqueeze(-1)
                .matmul(capital_u[link].unsqueeze(-2)) / d[link].reshape({batches, 1, 1});
            reduced_bias = bias_forces[link] + matvec(reduced_inertia, bias_accelerations[link]) +
                capital_u[link] * (u[link] / d[link]).unsqueeze(-1);
        }
        const auto parent = tree.parents[link];
        if (parent >= 0) {
            articulated[parent] = articulated[parent] + terms.transforms[link].transpose(-1, -2)
                .matmul(reduced_inertia).matmul(terms.transforms[link]);
            bias_forces[parent] = bias_forces[parent] +
                matvec(terms.transforms[link].transpose(-1, -2), reduced_bias);
        }
    }

    gravity = gravity.to(flat_q.options()).expand({batches, 3});
    auto base = torch::cat({torch::zeros_like(gravity), -gravity}, -1);
    std::vector<Tensor> accelerations;
    std::vector<Tensor> result(flat_q.size(1), torch::zeros({batches}, flat_q.options()));
    for (size_t link = 0; link < tree.parents.size(); ++link) {
        const auto parent = tree.parents[link];
        auto acceleration = (parent < 0 ? base : matvec(terms.transforms[link], accelerations[parent])) +
            bias_accelerations[link];
        const auto coordinate = tree.coordinates[link];
        if (coordinate >= 0) {
            result[coordinate] =
                (u[link] - batch_dot(capital_u[link], acceleration)) / d[link];
            acceleration = acceleration + terms.subspaces[link].unsqueeze(0) *
                result[coordinate].unsqueeze(-1);
        }
        accelerations.push_back(acceleration);
    }
    return torch::stack(result, -1).reshape(shape);
}

Tensor vector_meta(
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    Tensor q, Tensor, Tensor, Tensor, c10::optional<Tensor>) {
    return torch::empty_like(q, q.options().device(torch::kMeta));
}

Tensor matrix_meta(
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor q) {
    auto shape = q.sizes().vec();
    shape.push_back(q.size(-1));
    return torch::empty(shape, q.options().device(torch::kMeta));
}

Tensor aba_meta(
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    Tensor q, Tensor, Tensor, Tensor) {
    return torch::empty_like(q, q.options().device(torch::kMeta));
}

TORCH_LIBRARY_FRAGMENT(mechanica, m) {
    m.def("rnea(Tensor parents, Tensor joint_types, Tensor q_indices, Tensor axes, Tensor origins, Tensor multipliers, Tensor offsets, Tensor masses, Tensor centers, Tensor inertias, Tensor q, Tensor qdot, Tensor qddot, Tensor gravity, Tensor? external_forces) -> Tensor");
    m.def("crba(Tensor parents, Tensor joint_types, Tensor q_indices, Tensor axes, Tensor origins, Tensor multipliers, Tensor offsets, Tensor masses, Tensor centers, Tensor inertias, Tensor q) -> Tensor");
    m.def("aba(Tensor parents, Tensor joint_types, Tensor q_indices, Tensor axes, Tensor origins, Tensor multipliers, Tensor offsets, Tensor masses, Tensor centers, Tensor inertias, Tensor q, Tensor qdot, Tensor generalized_forces, Tensor gravity) -> Tensor");
}

TORCH_LIBRARY_IMPL(mechanica, CompositeExplicitAutograd, m) {
    m.impl("rnea", &native_rnea);
    m.impl("crba", &native_crba);
    m.impl("aba", &native_aba);
}

TORCH_LIBRARY_IMPL(mechanica, Autograd, m) {
    m.impl("rnea", &native_rnea);
    m.impl("crba", &native_crba);
    m.impl("aba", &native_aba);
}

TORCH_LIBRARY_IMPL(mechanica, Meta, m) {
    m.impl("rnea", &vector_meta);
    m.impl("crba", &matrix_meta);
    m.impl("aba", &aba_meta);
}
