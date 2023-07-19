# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/Users/jmarti53/Documents/Projects/LINKPROJECT/ELCA/cpp/process_hclust/build/temp.macosx-12-x86_64-cpython-39/process_hclust/_deps/pybind11-src"
  "/Users/jmarti53/Documents/Projects/LINKPROJECT/ELCA/cpp/process_hclust/build/temp.macosx-12-x86_64-cpython-39/process_hclust/_deps/pybind11-build"
  "/Users/jmarti53/Documents/Projects/LINKPROJECT/ELCA/cpp/process_hclust/build/temp.macosx-12-x86_64-cpython-39/process_hclust/_deps/pybind11-subbuild/pybind11-populate-prefix"
  "/Users/jmarti53/Documents/Projects/LINKPROJECT/ELCA/cpp/process_hclust/build/temp.macosx-12-x86_64-cpython-39/process_hclust/_deps/pybind11-subbuild/pybind11-populate-prefix/tmp"
  "/Users/jmarti53/Documents/Projects/LINKPROJECT/ELCA/cpp/process_hclust/build/temp.macosx-12-x86_64-cpython-39/process_hclust/_deps/pybind11-subbuild/pybind11-populate-prefix/src/pybind11-populate-stamp"
  "/Users/jmarti53/Documents/Projects/LINKPROJECT/ELCA/cpp/process_hclust/build/temp.macosx-12-x86_64-cpython-39/process_hclust/_deps/pybind11-subbuild/pybind11-populate-prefix/src"
  "/Users/jmarti53/Documents/Projects/LINKPROJECT/ELCA/cpp/process_hclust/build/temp.macosx-12-x86_64-cpython-39/process_hclust/_deps/pybind11-subbuild/pybind11-populate-prefix/src/pybind11-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/jmarti53/Documents/Projects/LINKPROJECT/ELCA/cpp/process_hclust/build/temp.macosx-12-x86_64-cpython-39/process_hclust/_deps/pybind11-subbuild/pybind11-populate-prefix/src/pybind11-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/jmarti53/Documents/Projects/LINKPROJECT/ELCA/cpp/process_hclust/build/temp.macosx-12-x86_64-cpython-39/process_hclust/_deps/pybind11-subbuild/pybind11-populate-prefix/src/pybind11-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
