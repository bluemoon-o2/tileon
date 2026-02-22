#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_ir(py::module_ &&m);
void init_interpreter(py::module_ &&m);
void init_specialize(py::module_ &m);
void init_env(py::module_ &m);

PYBIND11_MODULE(_C, m) {
  m.doc() = "Tileon C++ API";

  init_env(m);
  init_ir(m.def_submodule("ir", "Tilen IR API"));
  init_interpreter(m.def_submodule("interpreter", "Tilen Interpreter API"));
  init_specialize(m);
}
