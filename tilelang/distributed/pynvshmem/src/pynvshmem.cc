#include "c10/cuda/CUDAGuard.h"
#include <ATen/ops/from_blob.h>
#include <bootstrap_device_host/nvshmem_uniqueid.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <nvshmemx.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <torch/all.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/python.h>

class LazyLogger {
public:
  LazyLogger(bool no_error = false) {
    _no_print = no_error;
    _no_error = no_error;
  };

  ~LazyLogger() {
    if (!_no_print) {
      std::cerr << _message.str() << std::endl;
    }
    if (!_no_error) {
      throw std::runtime_error(_message.str());
    }
  }

  template <typename T> LazyLogger &operator<<(const T &value) {
    _message << value;
    return *this;
  }

private:
  bool _no_print = false;
  bool _no_error = false;
  std::ostringstream _message;
};

#define CUDA_CHECK(cuda_error)                                                 \
  {                                                                            \
    if (cuda_error != cudaSuccess) {                                           \
      printf("cudaError %s in %s:%d\n", cudaGetErrorString(cuda_error),        \
             __func__, __LINE__);                                              \
      throw std::runtime_error("cuda error.");                                 \
    }                                                                          \
  }

#define PYNVSHMEM_CHECK(cond)                                                  \
  LazyLogger(cond) << __FILE__ << ":" << __LINE__                              \
                   << " Check failed: " #cond ". "
#define PYNVSHMEM_CHECK_NE(a, b) PYNVSHMEM_CHECK(((a) != (b)))

#define CHECK_NVSHMEMX(expr)                                                   \
  do {                                                                         \
    int x = expr;                                                              \
    if (x != NVSHMEMX_SUCCESS) {                                               \
      throw std::runtime_error(__FILE__ ":" + std::to_string(__LINE__) +       \
                               " " #expr " failed with status code " +         \
                               std::to_string(x));                             \
    }                                                                          \
  } while (0)

namespace {
std::array<const char *, 5> kNvshmemInitStatus = {
    "NVSHMEM_STATUS_NOT_INITIALIZED", "NVSHMEM_STATUS_IS_BOOTSTRAPPED",
    "NVSHMEM_STATUS_IS_INITIALIZED", "NVSHMEM_STATUS_LIMITED_MPG",
    "NVSHMEM_STATUS_FULL_MPG"};
void check_nvshmem_init() {
  CHECK(nvshmemx_init_status() >= NVSHMEM_STATUS_IS_INITIALIZED);
}
} // namespace

#define NVSHMEMI_TYPENAME_P_IMPL_PYBIND(TYPENAME, TYPE)                        \
  void TYPENAME##_p(ptrdiff_t ptr, TYPE value, int peer) {                     \
    check_nvshmem_init();                                                      \
    nvshmem_##TYPENAME##_p((TYPE *)ptr, value, peer);                          \
  }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_TYPENAME_P_IMPL_PYBIND)
#undef NVSHMEMI_TYPENAME_P_IMPL_PYBIND

inline torch::Tensor 
nvshmem_create_tensor(const std::vector<int64_t> &shape,
                                   c10::ScalarType dtype) {
  check_nvshmem_init();
  auto current_device = c10::cuda::current_device();
  auto option_gpu =
      at::TensorOptions(at::kCUDA).dtype(dtype).device_index(current_device);
  auto size = torch::elementSize(dtype) *
              std::accumulate(shape.begin(), shape.end(), (size_t)1,
                              std::multiplies<>());
  PYNVSHMEM_CHECK_NE(size, 0);
  at::cuda::device_synchronize();
  void *ptr = nvshmem_malloc(size);
  CHECK(ptr != nullptr) << " nvshmem_malloc failed for malloc " << size;
  return at::from_blob(
          ptr, shape,
          [=](void *ptr) {
            // std::cerr << "enter nvshmem_free "
            // << ptr << "\n";
            at::cuda::CUDAGuard guard(current_device);
            at::cuda::device_synchronize();
            // std::cerr << "do nvshmem_free " <<
            // ptr << "\n";
            nvshmem_free(ptr);
            at::cuda::device_synchronize();
            // std::cerr << "exit nvshmem_free "
            // << ptr << "\n";
          },
          option_gpu);
}

std::vector<torch::Tensor>
nvshmem_create_tensor_list_intra_node(const std::vector<int64_t> &shape,
                           c10::ScalarType dtype) {
  check_nvshmem_init();
  auto current_device = c10::cuda::current_device();
  auto option_gpu =
      at::TensorOptions(at::kCUDA).dtype(dtype).device_index(current_device);
  auto size = torch::elementSize(dtype) *
              std::accumulate(shape.begin(), shape.end(), (size_t)1,
                              std::multiplies<>());
  PYNVSHMEM_CHECK_NE(size, 0);
  int local_world_size = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
  int rank = nvshmem_my_pe();
  int local_rank = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  std::vector<torch::Tensor> tensors;
  tensors.reserve(local_world_size);
  // std::cerr << "enter nvshmem_malloc\n";
  at::cuda::device_synchronize();
  // std::cerr << "do nvshmem_malloc\n";
  void *ptr = nvshmem_malloc(size);
  // std::cerr << "exit nvshmem_malloc " << ptr << "\n";

  // CUDA_CHECK(cudaMemset(ptr, 0, size)); // memset the allocated buffer
  CHECK(ptr != nullptr) << " nvshmem_malloc failed for malloc " << size;
  int rank_offset = rank - local_rank;
  for (int i = 0; i < local_world_size; i++) {
    // runs this call nvshmem failure, don't know why
    //  nvshmem_team_translate_pe(NVSHMEMX_TEAM_NODE, local_rank,
    //  NVSHMEM_TEAM_WORLD)
    int rank_global = i + rank_offset;
    if (rank == rank_global) {
      tensors.emplace_back(at::from_blob(
          ptr, shape,
          [=](void *ptr) {
            // std::cerr << "enter nvshmem_free "
            // << ptr << "\n";
            at::cuda::CUDAGuard guard(current_device);
            at::cuda::device_synchronize();
            // std::cerr << "do nvshmem_free " <<
            // ptr << "\n";
            nvshmem_free(ptr);
            at::cuda::device_synchronize();
            // std::cerr << "exit nvshmem_free "
            // << ptr << "\n";
          },
          option_gpu));
    } else {
      void *rptr = nvshmem_ptr(ptr, rank_global);
      PYNVSHMEM_CHECK(rptr != nullptr) << "rank " << rank;
      tensors.emplace_back(at::from_blob(rptr, shape, option_gpu));
    }
  }

  return tensors;
}


PYBIND11_MODULE(_pynvshmem, m) {
  /* Basic queries */
  m.def("nvshmem_my_pe", []() -> int {
    check_nvshmem_init();
    return nvshmem_my_pe();
  });
  m.def("nvshmem_n_pes", []() -> int {
    check_nvshmem_init();
    return nvshmem_n_pes();
  });
  m.def("nvshmem_team_my_pe", [](int team) {
    check_nvshmem_init();
    return nvshmem_team_my_pe(team);
  });
  m.def("nvshmem_team_n_pes", [](int team) {
    check_nvshmem_init();
    return nvshmem_team_n_pes(team);
  });

  /* CUmodule related */
  m.def("nvshmemx_cumodule_init", [](intptr_t module) {
    CHECK_NVSHMEMX(nvshmemx_cumodule_init((CUmodule)module));
  });
  m.def("nvshmemx_cumodule_finalize", [](intptr_t module) {
    CHECK_NVSHMEMX(nvshmemx_cumodule_finalize((CUmodule)module));
  });

  /* Memory related */
  m.def("nvshmem_malloc", [](size_t size) {
    void *ptr = nvshmem_malloc(size);
    if (ptr == nullptr) {
      throw std::runtime_error("nvshmem_malloc failed");
    }
    return (intptr_t)ptr;
  });
  m.def("nvshmem_free", [](intptr_t ptr) {
    check_nvshmem_init();
    nvshmem_free((void *)ptr);
  });
  m.def("nvshmem_ptr", [](intptr_t ptr, int peer) {
    return (intptr_t)nvshmem_ptr((void *)ptr, peer);
  });
  m.def("nvshmemx_mc_ptr", [](nvshmemx_team_t team, intptr_t ptr) {
    return (intptr_t)nvshmemx_mc_ptr(team, (void *)ptr);
  });

  /* Unique ID related */
  m.def("nvshmemx_get_uniqueid", []() {
    nvshmemx_uniqueid_t id;
    CHECK_NVSHMEMX(nvshmemx_get_uniqueid(&id));
    std::string bytes((char *)&id, sizeof(id));
    return pybind11::bytes(bytes);
  });
  m.def("nvshmemx_init_attr_with_uniqueid", [](int rank, int nranks,
                                               pybind11::bytes bytes) {
    nvshmemx_uniqueid_t id;
    std::string id_str = bytes;
    if (id_str.size() != sizeof(id)) {
      throw std::runtime_error(
          "nvshmemx_init_attr_with_uniqueid: invalid size");
    }
    nvshmemx_init_attr_t init_attr;
    CHECK_NVSHMEMX(
        nvshmemx_set_attr_uniqueid_args(rank, nranks, &id, &init_attr));
    memcpy(&id, id_str.data(), sizeof(id));
    CHECK_NVSHMEMX(nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &init_attr));
  });

  /* Single-element put for all dtypes */
#define NVSHMEMI_TYPENAME_P_PYBIND(TYPENAME, TYPE)                             \
  m.def("nvshmem_" #TYPENAME "_p", &TYPENAME##_p);
  NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_TYPENAME_P_PYBIND)
#undef NVSHMEMI_TYPENAME_P_PYBIND

  /* Barrier related */
  m.def("nvshmem_barrier_all", []() {
    check_nvshmem_init();
    nvshmem_barrier_all();
  });
  m.def("nvshmemx_barrier_all_on_stream", [](intptr_t stream) {
    nvshmemx_barrier_all_on_stream((cudaStream_t)stream);
  });

  /* Tensor creation */
  m.def("nvshmem_create_tensor",
        [](const std::vector<int64_t> shape, py::object dtype) {
          auto cast_dtype = torch::python::detail::py_object_to_dtype(dtype);
          return nvshmem_create_tensor(shape, cast_dtype);
        });
  m.def(
      "nvshmem_create_tensor_list_intra_node",
      [](const std::vector<int64_t> &shape, py::object dtype) {
        return nvshmem_create_tensor_list_intra_node(
            shape, torch::python::detail::py_object_to_dtype(std::move(dtype)));
      },
      py::arg("shape"), py::arg("dtype"));

  /* RMA */
  m.def("nvshmem_putmem",
        [](intptr_t dest, const intptr_t source, size_t nelems, int pe) {
          check_nvshmem_init();
          nvshmem_putmem((void *)dest, (const void *)source, nelems, pe);
        });
  m.def("nvshmem_getmem",
        [](intptr_t dest, const intptr_t source, size_t nelems, int pe) {
          check_nvshmem_init();
          nvshmem_getmem((void *)dest, (const void *)source, nelems, pe);
        });

  m.def("nvshmemx_putmem_on_stream",
        [](intptr_t dest, const intptr_t source, size_t nelems, int pe,
           intptr_t stream) {
          check_nvshmem_init();
          nvshmemx_putmem_on_stream((void *)dest, (const void *)source, nelems,
                                    pe, (cudaStream_t)stream);
        });
  m.def("nvshmemx_getmem_on_stream",
        [](intptr_t dest, const intptr_t source, size_t nelems, int pe,
           intptr_t stream) {
          check_nvshmem_init();
          nvshmemx_getmem_on_stream((void *)dest, (const void *)source, nelems,
                                    pe, (cudaStream_t)stream);
        });
  m.def("nvshmemx_putmem_signal_on_stream",
        [](intptr_t dest, const intptr_t source, size_t nelems,
           intptr_t sig_addr, uint64_t signal, int sig_op, int pe,
           intptr_t stream) {
          check_nvshmem_init();
          nvshmemx_putmem_signal_on_stream((void *)dest, (const void *)source,
                                           nelems, (uint64_t *)sig_addr, signal,
                                           sig_op, pe, (cudaStream_t)stream);
        });
}
