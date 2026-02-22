#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <map>
#include <optional>
#include <set>
#include <string>

namespace py = pybind11;
using namespace pybind11::literals;

namespace Env {

// Helper function to check if a string is truthy
// Returns true if the string is "1", "y", "on", "yes", or "true" (case-insensitive)
bool is_truthy(const char *str) {
  if (!str) return false;
  
  std::string s(str);
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  return s == "1" || s == "y" || s == "on" || s == "yes" || s == "true";
}

// Get an environment variable
// If the variable is not set, return the default value (None by default)
py::object getenv(const std::string &name, py::object default_val = py::none()) {
  const char *env_val = std::getenv(name.c_str());
  if (!env_val) {
    return default_val;
  }
  return py::str(env_val);
}

// Get an environment variable as a boolean
// If the variable is set, return True if it is truthy, False otherwise
// If the variable is not set, return the default value
py::object getenv_bool(const std::string &name, py::object default_val) {
  const char *env_val = std::getenv(name.c_str());
  if (env_val) {
    return py::bool_(is_truthy(env_val));
  }
  return default_val;
}

// List of environment variables that invalidate the cache
const std::set<std::string> CACHE_INVALIDATING_ENV_VARS = {
    "AMDGCN_ENABLE_DUMP",
    "AMDGCN_USE_BUFFER_ATOMICS",
    "AMDGCN_USE_BUFFER_OPS",
    "DISABLE_LLVM_OPT",
    "DISABLE_MMA_V3",
    "DISABLE_MMA_V5",
    "DISABLE_PTXAS_OPT",
    "LLVM_IR_ENABLE_DUMP",
    "LLVM_ENABLE_TIMING",
    "LLVM_PASS_PLUGIN_PATH",
    "LLVM_EXTRACT_DI_LOCAL_VARIABLES",
    "MLIR_ENABLE_DIAGNOSTICS",
    "MLIR_ENABLE_DUMP",
    "MLIR_DUMP_PATH",
    "MLIR_ENABLE_TIMING",
    "MLIR_DISABLE_MULTITHREADING",
    "TILEON_DEFAULT_FP_FUSION",
    "TILEON_DISABLE_LINE_INFO",
    "TILEON_DUMP_MIR",
    "TILEON_ENABLE_LLVM_DEBUG",
    "TILEON_HIP_USE_ASYNC_COPY",
    "TILEON_HIP_USE_BLOCK_PINGPONG",
    "TILEON_HIP_USE_IN_THREAD_TRANSPOSE",
    "TILEON_LLVM_DEBUG_ONLY",
    "TILEON_ENABLE_ASAN",
    "TILEON_OVERRIDE_ARCH",
    "USE_IR_LOC",
    "NVPTX_ENABLE_DUMP",
    "ALLOW_LHS_TMEM_LAYOUT_CONVERSION",
    "TILEON_F32_DEFAULT",
    "TILEON_PREFER_TMEM_16x256_LAYOUT",
    "TILEON_ENABLE_EXPERIMENTAL_CONSAN",
    "TILEON_PASS_PLUGIN_PATH",
    "TILEON_PARTITION_SCHEDULING_ENABLE_DUMP_DOT",
    "TILEON_PARTITION_SCHEDULING_DUMP_DATA_ONLY",
    "TILEON_PARTITION_SCHEDULING_DUMP_LOOP_ONLY",
};

// Get a map of cache-invalidating environment variables and their values
// The values are normalized to "true"/"false" for boolean-like variables
std::map<std::string, std::string> get_cache_invalidating_env_vars() {
  static std::map<std::string, std::string> cache;
  static bool initialized = false;

  if (initialized) {
     return cache;
  }

  for (const auto &envVar : CACHE_INVALIDATING_ENV_VARS) {
    const char *cstr = std::getenv(envVar.c_str());
    if (!cstr)
      continue;
    std::string strVal(cstr);
    if (strVal.empty())
      continue;

    // Check if bool and normalize
    std::string lowerStr = strVal;
    std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    
    if (lowerStr == "on" || lowerStr == "true" || lowerStr == "1")
      cache[envVar] = "true";
    else if (lowerStr == "off" || lowerStr == "false" || lowerStr == "0")
      cache[envVar] = "false";
    else
      cache[envVar] = strVal;
  }
  initialized = true;
  return cache;
}

} // namespace Env

using namespace Env;

void init_env(py::module_ &m) {
  // Bind env var functions with detailed documentation
  
  m.def("getenv", &Env::getenv, "name"_a, "default_val"_a = py::none(),
        R"pbdoc(
    Get an environment variable.

    Args:
        name (str): The name of the environment variable.
        default_val (object, optional): The value to return if the environment variable is not set. Defaults to None.

    Returns:
        str or object: The value of the environment variable, or default_val if not set.
    )pbdoc");

  m.def("getenv_bool", &Env::getenv_bool, "name"_a, "default_val"_a,
        R"pbdoc(
    Get an environment variable as a boolean.

    Args:
        name (str): The name of the environment variable.
        default_val (object): The value to return if the environment variable is not set.

    Returns:
        bool or object: True if the environment variable is set to a truthy value ('1', 'y', 'on', 'yes', 'true'),
                        False if it is set to a falsy value, or default_val if not set.
    )pbdoc");

  m.def("get_cache_invalidating_env_vars", &Env::get_cache_invalidating_env_vars,
        R"pbdoc(
    Get a dictionary of environment variables that affect compilation cache.

    Returns:
        dict: A dictionary mapping environment variable names to their values.
              Boolean-like values are normalized to "true" or "false".
    )pbdoc");
}
