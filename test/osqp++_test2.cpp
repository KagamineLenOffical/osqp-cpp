#include <gflags/gflags.h>
#include <osqp++.h>
#include <igl/canonical_quaternions.h>
#include <igl/xml/serialize_xml.h>
DEFINE_string(filename_prefix,
"", "common prefix for an instance of data, default empty");

void settingsinit(osqp::OsqpSettings &settings) {
    settings.eps_rel = 1e-4;
    settings.max_iter = 100000;
    settings.eps_prim_inf = 1e-6;
    settings.eps_dual_inf = 1e-6;
    settings.adaptive_rho = true;
    settings.warm_start = true;
}

int main(int argc, char *argv[]) {
    using google::ParseCommandLineFlags;
    ParseCommandLineFlags(&argc, &argv, true);
    Eigen::SparseMatrix<double> constrains, lhs;
    std::string prefix = FLAGS_filename_prefix;
    std::cout << "common prefix '"
              << prefix << "'\n";
    igl::xml::deserialize_xml(constrains, "constrains", prefix + "constrain.xml");
    igl::xml::deserialize_xml(lhs, "lhs", prefix + "lhs.xml");
    Eigen::MatrixXd lb, ub, rhs, result;
    igl::xml::deserialize_xml(lb, "lb", prefix + "lb.xml");
    igl::xml::deserialize_xml(ub, "ub", prefix + "ub.xml");
    igl::xml::deserialize_xml(rhs, "rhs", prefix + "rhs.xml");
    igl::xml::deserialize_xml(result, "result", prefix + "result.xml");
    osqp::OsqpInstance instance;
    osqp::OsqpSettings settings;
    osqp::OsqpSolver solver;
    settingsinit(settings);
    instance.objective_matrix = -lhs;
    instance.constraint_matrix = constrains;
    instance.lower_bounds = lb;
    instance.upper_bounds = ub;
    instance.objective_vector = Eigen::VectorXd(rhs);
    auto status = solver.Init(instance, settings);
    std::cout << "solver init status "
              << status << std::endl;
    osqp::OsqpExitCode exit_code = solver.Solve();
    std::cout << "solver solve status "
              << ToString(exit_code) << std::endl;
    double optimal_objective = solver.objective_value();
    Eigen::MatrixXd optimal_solution = solver.primal_solution();
    std::cout << "Relative difference between pre = "
              << (optimal_solution - result).norm() << std::endl;
}