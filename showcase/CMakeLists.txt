# Speed of sound for two equation-of-state modules of the MUSES framework, see showcase.ipynb
foreach(name njl cmf)
    add_executable(showcase_${name} ${name}.cpp)
    target_link_libraries(showcase_${name} PRIVATE mfderiv::mfderiv)
endforeach()
