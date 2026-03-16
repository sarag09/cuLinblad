#include <iostream>
#include "culindblad/types.hpp"

int main()
{
    using namespace culindblad;

    std::cout << "cuLindblad core solver starting..." << std::endl;

    Real r = 1.23;
    Complex z = {2.0, -1.0};
    Index n = 42;

    std::cout << "Real example: " << r << std::endl;
    std::cout << "Complex example: " << z << std::endl;
    std::cout << "Index example: " << n << std::endl;

    return 0;
}
