#include <exception>
#include <iostream>
#include <string>

#include "orchard_solver/discretization/Assembler.h"
#include "orchard_solver/io/ModelIO.h"
#include "orchard_solver/solver_core/DynamicSystem.h"

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 3) {
            std::cerr << "Usage: orchard_cli <model.json> [output.csv]\n";
            return 1;
        }

        const std::string model_path = argv[1];
        const orchard::OrchardModel model = orchard::loadModelFromFile(model_path);

        orchard::StructuralAssembler assembler;
        const orchard::AssembledModel assembled = assembler.assemble(model);
        const std::string output_path = argc == 3 ? argv[2] : model.analysis.output_csv;

        std::cout << "Model: " << model.metadata.name << '\n';
        std::cout << "Branches: " << model.branches.size() << ", Fruits: " << model.fruits.size() << '\n';
        std::cout << "DOFs: " << assembled.system.dof_labels.size() << '\n';
        std::cout << "Nonlinear links: " << assembled.system.nonlinear_links.size() << '\n';

        if (model.analysis.mode == orchard::AnalysisMode::FrequencyResponse) {
            orchard::FrequencyResponseAnalyzer analyzer;
            const orchard::FrequencyResponseResult response = analyzer.analyze(
                assembled.system,
                assembled.excitation_dof,
                model.excitation,
                model.analysis,
                assembled.observation_names,
                assembled.observation_dofs
            );
            response.writeCsv(output_path);
            std::cout << "Analysis mode: frequency_response" << '\n';
            std::cout << "Frequency steps: " << response.points.size() << '\n';
        } else {
            orchard::NewmarkIntegrator integrator;
            const orchard::TimeHistoryResult response = integrator.analyze(
                assembled.system,
                assembled.excitation_dof,
                model.excitation,
                model.analysis,
                assembled.observation_names,
                assembled.observation_dofs
            );
            response.writeCsv(output_path);
            std::cout << "Analysis mode: time_history" << '\n';
            std::cout << "Time samples: " << response.points.size() << '\n';
        }

        std::cout << "Output: " << output_path << '\n';

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "orchard_cli failed: " << ex.what() << '\n';
        return 1;
    }
}
