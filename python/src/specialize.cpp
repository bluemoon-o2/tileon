#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <unordered_map>
#include <utility>

namespace py = pybind11;
using namespace pybind11::literals;

// Use a distinct namespace to avoid collisions
namespace specialize {

// -----------------------------------------------------------------------------
// Type Definitions & Globals
// -----------------------------------------------------------------------------

using DTypePtrKey = std::pair<Py_hash_t, bool>;
using DTypeKey = Py_hash_t;

struct DTypePtrKeyHash {
  std::size_t operator()(const DTypePtrKey &k) const {
    // Combine hash of pointer/hash and bool
    return std::hash<Py_hash_t>()(k.first) ^ (std::hash<bool>()(k.second) << 1);
  }
};

using DtypePtr2Str =
    std::unordered_map<DTypePtrKey, py::object, DTypePtrKeyHash>;
using Dtype2Str = std::unordered_map<DTypeKey, py::object>;

// Handler signature using py::handle for efficiency (no refcount churn)
using TypeHandler = std::pair<py::object, py::object> (*)(py::handle,
                                                          py::handle, bool,
                                                          bool, bool);
using TypeHandlerCache = std::unordered_map<PyTypeObject *, TypeHandler>;

// Forward declarations
static std::pair<py::object, py::object>
specialize_arg(py::handle backend, py::handle arg, bool is_const,
               bool specialize_value, bool align);

// Globals - we use raw PyObject* to avoid destruction order issues at exit,
// but wrap them in py::handle/py::object when using.
static bool init_called = false;

static PyObject *constexpr_cls = nullptr;
static PyObject *jit_callable_cls = nullptr;
static PyObject *tensor_descriptor_cls = nullptr;
static PyObject *canonicalize_dtype_fn = nullptr;
static PyObject *canonicalize_ptr_dtype_fn = nullptr;
static PyObject *torch_tensor_cls = nullptr;

// Interned strings
static PyObject *i32_str = nullptr;
static PyObject *i64_str = nullptr;
static PyObject *u64_str = nullptr;
static PyObject *fp32_str = nullptr;
static PyObject *u1_str = nullptr;
static PyObject *D_str = nullptr;
static PyObject *constexpr_str = nullptr;
static PyObject *empty_str = nullptr;

static PyObject *base_attr = nullptr;
static PyObject *data_ptr_attr = nullptr;
static PyObject *dtype_attr = nullptr;
static PyObject *cache_key_attr = nullptr;
static PyObject *_fields_attr = nullptr;
static PyObject *block_shape_attr = nullptr;
static PyObject *shape_attr = nullptr;
static PyObject *layout_attr = nullptr;
static PyObject *has_native_tensor_spec_attr = nullptr;
static PyObject *get_tensor_spec_attr = nullptr;
static PyObject *align_kwarg = nullptr;

// Caches
static DtypePtr2Str dtype_ptr2str;
static Dtype2Str dtype2str;
static TypeHandlerCache type_handler_cache;

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

PyObject *intern_from_string(const char *str) {
  PyObject *obj = PyUnicode_InternFromString(str);
  if (!obj)
    throw py::error_already_set();
  return obj;
}

PyObject *import_from(const char *module_name, const char *var_name) {
  py::object var = py::module_::import(module_name).attr(var_name);
  return var.release().ptr();
}

void init_interned_strings() {
  i32_str = intern_from_string("i32");
  i64_str = intern_from_string("i64");
  u64_str = intern_from_string("u64");
  fp32_str = intern_from_string("fp32");
  u1_str = intern_from_string("u1");
  D_str = intern_from_string("D");
  constexpr_str = intern_from_string("constexpr");
  empty_str = intern_from_string("");

  base_attr = intern_from_string("base");
  data_ptr_attr = intern_from_string("data_ptr");
  dtype_attr = intern_from_string("dtype");
  cache_key_attr = intern_from_string("cache_key");
  _fields_attr = intern_from_string("_fields");
  block_shape_attr = intern_from_string("block_shape");
  shape_attr = intern_from_string("shape");
  layout_attr = intern_from_string("layout");
  has_native_tensor_spec_attr =
      intern_from_string("supports_native_tensor_specialization");
  get_tensor_spec_attr = intern_from_string("get_tensor_specialization");

  align_kwarg = py::make_tuple("align").release().ptr();
}

// -----------------------------------------------------------------------------
// Specialize Handlers
// -----------------------------------------------------------------------------

std::pair<py::object, py::object> specialize_tensordesc(py::handle arg,
                                                        bool has_layout) {
  // Use py::handle::attr with interned strings for speed
  py::handle base = arg.attr(py::handle(base_attr));
  if (!base)
    return {};

  py::handle dtype = base.attr(py::handle(dtype_attr));
  if (!dtype)
    return {};

  py::object type_str_obj;
  Py_hash_t dtype_hash = py::hash(dtype);
  if (dtype_hash == -1)
    return {};

  DTypeKey dsk{dtype_hash};
  auto it = dtype2str.find(dsk);
  if (it != dtype2str.end()) {
    type_str_obj = it->second;
  } else {
    // Call canonicalize_dtype(dtype)
    py::object res =
        py::reinterpret_steal<py::object>(PyObject_CallFunctionObjArgs(
            canonicalize_dtype_fn, dtype.ptr(), nullptr));
    if (!res)
      return {};
    dtype2str[dsk] = res;
    type_str_obj = res;
  }

  std::string desc_cstr;
  desc_cstr.reserve(128);

  // Determine im2col by class type (Gluon only).
  bool is_im2col = false;
  desc_cstr = is_im2col ? "tensordesc_im2col<" : "tensordesc<";

  // Convert type_str to string representation
  py::object dtype_str = py::str(type_str_obj);
  desc_cstr += dtype_str.cast<std::string>();

  py::handle block_shape_obj = arg.attr(py::handle(block_shape_attr));
  if (!block_shape_obj)
    return {};

  // Convert block_shape to list then string
  py::object block_shape_list =
      py::reinterpret_steal<py::object>(PySequence_List(block_shape_obj.ptr()));
  if (!block_shape_list)
    return {};
  py::object block_shape_str = py::str(block_shape_list);
  desc_cstr += block_shape_str.cast<std::string>();

  // For im2col mode, append input tensor rank after block_shape
  if (is_im2col) {
    py::handle tensor_shape_obj = arg.attr(py::handle(shape_attr));
    if (!tensor_shape_obj)
      return {};
    Py_ssize_t tensor_rank = py::len(tensor_shape_obj);
    desc_cstr += ",input_rank=";
    desc_cstr += std::to_string(tensor_rank);
  }

  if (has_layout) {
    py::handle layout_obj = arg.attr(py::handle(layout_attr));
    if (!layout_obj)
      return {};
    py::object layout_repr = py::repr(layout_obj);
    desc_cstr += ",";
    desc_cstr += layout_repr.cast<std::string>();
  }

  desc_cstr += ">";
  return {py::str(desc_cstr), py::none()};
}

std::pair<py::object, py::object>
handle_long_type(py::handle backend, py::handle arg, bool is_const,
                 bool specialize_value, bool align) {
  int overflow;
  long long val = PyLong_AsLongLongAndOverflow(arg.ptr(), &overflow);
  if (PyErr_Occurred())
    return {};

  if (specialize_value && (val == 1)) {
    return {py::reinterpret_borrow<py::object>(constexpr_str),
            py::reinterpret_borrow<py::object>(arg)};
  }

  py::object type_str;
  py::object key_obj;
  if (overflow == 0) {
    type_str = py::reinterpret_borrow<py::object>(
        (val >= INT32_MIN && val <= INT32_MAX) ? i32_str : i64_str);
    if (specialize_value) {
      key_obj = py::reinterpret_borrow<py::object>(
          (align && ((val & 15) == 0)) ? D_str : empty_str);
    }
  } else {
    // Check unsigned long long
    unsigned long long val_64 = PyLong_AsUnsignedLongLong(arg.ptr());
    if (PyErr_Occurred()) {
      PyErr_SetString(PyExc_OverflowError,
                      "integer to be specialized too large to represent");
      return {};
    }
    type_str = py::reinterpret_borrow<py::object>(u64_str);
    if (specialize_value) {
      key_obj = py::reinterpret_borrow<py::object>(
          (align && ((val_64 & 15) == 0)) ? D_str : empty_str);
    }
  }

  if (!key_obj) {
    return {type_str, py::none()};
  }
  return {type_str, key_obj};
}

std::pair<py::object, py::object> handle_tensor(py::handle backend,
                                                py::handle arg, bool is_const,
                                                bool specialize_value,
                                                bool align) {
  // handle type_str specialization of a tensor
  py::handle dtype = arg.attr(py::handle(dtype_attr));
  if (!dtype)
    return {};

  Py_hash_t dtype_hash = py::hash(dtype);
  if (dtype_hash == -1)
    return {};

  DTypePtrKey dsk{dtype_hash, is_const};
  auto it = dtype_ptr2str.find(dsk);

  py::object type_str;
  if (it != dtype_ptr2str.end()) {
    type_str = it->second;
  } else {
    // Call canonicalize_ptr_dtype(dtype, is_const)
    py::object canon_res = py::reinterpret_steal<py::object>(
        PyObject_CallFunctionObjArgs(canonicalize_ptr_dtype_fn, dtype.ptr(),
                                     is_const ? Py_True : Py_False, nullptr));
    if (!canon_res)
      return {};
    dtype_ptr2str[dsk] = canon_res;
    type_str = canon_res;
  }

  // handle alignment specialization of a tensor
  if (!specialize_value) {
    return {type_str, py::none()};
  }

  bool native_impl_available = false;
  py::handle native_spec_obj =
      backend.attr(py::handle(has_native_tensor_spec_attr));
  if (native_spec_obj) {
    native_impl_available = native_spec_obj.cast<bool>();
  } else {
    PyErr_Clear();
  }

  py::object key;
  if (native_impl_available) {
    // call arg.data_ptr()
    py::object data_ptr_result = py::reinterpret_steal<py::object>(
        PyObject_CallMethodNoArgs(arg.ptr(), data_ptr_attr));
    if (!data_ptr_result)
      return {};

    unsigned long long data_ptr =
        PyLong_AsUnsignedLongLong(data_ptr_result.ptr());
    if (PyErr_Occurred())
      return {};

    PyObject *key_ptr = (align && ((data_ptr & 15) == 0)) ? D_str : empty_str;
    key = py::reinterpret_borrow<py::object>(key_ptr);
  } else {
    // call backend.get_tensor_specialization(arg, align=True/False)
    PyObject *args[3] = {backend.ptr(), arg.ptr(), align ? Py_True : Py_False};
    PyObject *kwnames = align_kwarg;
    key = py::reinterpret_steal<py::object>(
        PyObject_VectorcallMethod(get_tensor_spec_attr, args, 2, kwnames));
    if (!key)
      return {};
  }

  return {type_str, key};
}

std::pair<py::object, py::object>
handle_bool_type(py::handle backend, py::handle arg, bool is_const,
                 bool specialize_value, bool align) {
  return {py::reinterpret_borrow<py::object>(u1_str), py::none()};
}

std::pair<py::object, py::object>
handle_float_type(py::handle backend, py::handle arg, bool is_const,
                  bool specialize_value, bool align) {
  return {py::reinterpret_borrow<py::object>(fp32_str), py::none()};
}

std::pair<py::object, py::object>
handle_tensor_descriptor(py::handle backend, py::handle arg, bool is_const,
                         bool specialize_value, bool align) {
  return specialize_tensordesc(arg, false);
}

std::pair<py::object, py::object>
handle_gluon_tensor_descriptor(py::handle backend, py::handle arg,
                               bool is_const, bool specialize_value,
                               bool align) {
  return specialize_tensordesc(arg, true);
}

std::pair<py::object, py::object>
handle_constexpr_type(py::handle backend, py::handle arg, bool is_const,
                      bool specialize_value, bool align) {
  return {py::reinterpret_borrow<py::object>(constexpr_str),
          py::reinterpret_borrow<py::object>(arg)};
}

std::pair<py::object, py::object>
handle_jit_callable(py::handle backend, py::handle arg, bool is_const,
                    bool specialize_value, bool align) {
  py::handle cache_key = arg.attr(py::handle(cache_key_attr));
  if (!cache_key)
    return {};
  return {py::reinterpret_borrow<py::object>(constexpr_str),
          py::reinterpret_borrow<py::object>(cache_key)};
}

std::pair<py::object, py::object> handle_tuple(py::handle backend,
                                               py::handle arg, bool is_const,
                                               bool specialize_value,
                                               bool align) {
  py::tuple tuple_arg = py::reinterpret_borrow<py::tuple>(arg);
  size_t size = tuple_arg.size();
  if (size == 0) {
    return {tuple_arg, tuple_arg};
  }

  bool is_namedtuple = py::hasattr(arg, py::handle(_fields_attr));

  py::tuple tys_tuple(size);
  py::tuple keys_tuple(size);

  for (size_t i = 0; i < size; ++i) {
    py::handle item = tuple_arg[i];
    auto [type, key] = specialize_arg(backend, item, false, true, true);
    if (!type || !key)
      return {};
    tys_tuple[i] = type;
    keys_tuple[i] = key;
  }

  if (is_namedtuple) {
    // Reconstruct namedtuple
    py::handle tuple_type((PyObject *)Py_TYPE(arg.ptr()));
    py::object new_tys = py::reinterpret_steal<py::object>(
        PyObject_CallObject(tuple_type.ptr(), tys_tuple.ptr()));
    if (!new_tys)
      return {};

    py::object new_keys = py::reinterpret_steal<py::object>(
        PyObject_CallObject(tuple_type.ptr(), keys_tuple.ptr()));
    if (!new_keys)
      return {};

    return {new_tys, new_keys};
  }

  return {tys_tuple, keys_tuple};
}

void init_type_handler_cache() {
  type_handler_cache[&PyLong_Type] = handle_long_type;
  type_handler_cache[&PyBool_Type] = handle_bool_type;
  type_handler_cache[&PyFloat_Type] = handle_float_type;
  type_handler_cache[&PyTuple_Type] = handle_tuple;

  // Custom Classes
  auto register_if_valid = [](PyObject *cls, TypeHandler handler) {
    if (cls && PyType_Check(cls)) {
      type_handler_cache[(PyTypeObject *)cls] = handler;
    }
  };

  register_if_valid(torch_tensor_cls, handle_tensor);
  register_if_valid(tensor_descriptor_cls, handle_tensor_descriptor);
  register_if_valid(constexpr_cls, handle_constexpr_type);
  register_if_valid(jit_callable_cls, handle_jit_callable);
}

// -----------------------------------------------------------------------------
// Initialization & Entry Point
// -----------------------------------------------------------------------------

bool init_globals() noexcept try {
  jit_callable_cls = import_from("tileon.runtime.jit", "JITCallable");
  tensor_descriptor_cls =
      import_from("tileon.runtime.interpreter", "TensorDescriptor");
  canonicalize_dtype_fn = import_from("tileon._utils", "canonicalize_dtype");
  canonicalize_ptr_dtype_fn =
      import_from("tileon._utils", "canonicalize_ptr_dtype");
  constexpr_cls = import_from("tileon.language", "constexpr");
  torch_tensor_cls = import_from("torch", "Tensor");

  init_interned_strings();
  init_type_handler_cache();

  init_called = true;
  return true;
} catch (py::error_already_set &e) {
  e.restore();
  return false;
}

std::pair<py::object, py::object> specialize_arg(py::handle backend,
                                                 py::handle arg, bool is_const,
                                                 bool specialize_value,
                                                 bool align) {
  // Fast path: check exact type in cache
  PyTypeObject *arg_type = Py_TYPE(arg.ptr());
  auto it = type_handler_cache.find(arg_type);
  if (it != type_handler_cache.end()) {
    return it->second(backend, arg, is_const, specialize_value, align);
  }

  if (arg.is_none()) {
    return {py::reinterpret_borrow<py::object>(constexpr_str), py::none()};
  }

  // Handle tuple subclasses (e.g. namedtuple)
  if (py::isinstance<py::tuple>(arg)) {
    return handle_tuple(backend, arg, is_const, specialize_value, align);
  }

  // Fallback paths with caching optimization
  auto check_and_cache = [&](PyObject *cls, TypeHandler handler) -> bool {
    if (cls && PyObject_IsInstance(arg.ptr(), cls) == 1) {
      type_handler_cache[arg_type] = handler;
      return true;
    }
    return false;
  };

  if (check_and_cache(constexpr_cls, handle_constexpr_type)) {
    return handle_constexpr_type(backend, arg, is_const, specialize_value,
                                 align);
  }
  if (check_and_cache(tensor_descriptor_cls, handle_tensor_descriptor)) {
    return handle_tensor_descriptor(backend, arg, is_const, specialize_value,
                                    align);
  }
  if (check_and_cache(jit_callable_cls, handle_jit_callable)) {
    return handle_jit_callable(backend, arg, is_const, specialize_value, align);
  }

  // Fallback checking attributes directly
  if (py::hasattr(arg, py::handle(data_ptr_attr))) {
    return handle_tensor(backend, arg, is_const, specialize_value, align);
  }

  //  Default types fallback
  if (py::isinstance<py::int_>(arg)) {
    return handle_long_type(backend, arg, is_const, specialize_value, align);
  }
  if (py::isinstance<py::float_>(arg)) {
    return handle_float_type(backend, arg, is_const, specialize_value, align);
  }

  return {};
}

py::tuple specialize_impl(py::object backend, py::object arg, bool is_const,
                          bool specialize_value, bool align) {
  if (!init_called) {
    if (!init_globals()) {
      throw py::error_already_set();
    }
  }

  auto [type, key] =
      specialize_arg(backend, arg, is_const, specialize_value, align);

  if (!type || !key) {
    if (!PyErr_Occurred()) {
      std::string type_name =
          py::type::of(arg).attr("__name__").cast<std::string>();
      throw py::type_error("failed to specialize argument of type: " +
                           type_name);
    }
    throw py::error_already_set();
  }
  return py::make_tuple(type, key);
}

} // namespace specialize

void init_specialize(py::module_ &m) {
  // Corrected argument names to match usage
  m.def("native_specialize_impl", &specialize::specialize_impl, "self"_a,
        "arg"_a, "is_const"_a, "specialize"_a, "align"_a,
        R"pbdoc(
        Specialize an argument.

        Args:
            self: The backend instance.
            arg: The argument to specialize.
            is_const: Whether the argument is constant.
            specialize: Whether to specialize on value.
            align: Whether to align.

        Returns:
            A tuple of (type, key) where type is the specialized type and key is the specialized key.
        )pbdoc");
}
