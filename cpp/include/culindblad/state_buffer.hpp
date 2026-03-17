#pragma once

#include <cstddef>

#include "culindblad/types.hpp"

namespace culindblad {

struct StateBuffer {
    Complex* data;
    Index size;

    Complex& operator[](Index i) {
        return data[i];
    }

    const Complex& operator[](Index i) const {
        return data[i];
    }
};

struct ConstStateBuffer {
    const Complex* data;
    Index size;

    const Complex& operator[](Index i) const {
        return data[i];
    }
};

} // namespace culindblad
