#include <pybind11/pybind11.h>
#include <omp/HandEvaluator.h>
#include <pybind11/stl.h>
#include <omp/lbr.h>

namespace py = pybind11;

int evaluate_hand(const std::vector<int>& cardints) {
    omp::HandEvaluator eval;
    omp::Hand h = omp::Hand::empty();
    for (int cardint : cardints) {
        h += omp::Hand(cardint);
    }
    return eval.evaluate(h);
}

PYBIND11_MODULE(ompeval, m) {
    m.doc() = "Python bindings for OMPEval";
    m.def("evaluate_hand", &evaluate_hand, "Evaluate a poker hand");
    py::class_<omp::LBR>(m, "LBR")
        .def(py::init<>())
        .def("wprollout", &omp::LBR::wprollout, "Compute win probability rollout");
}