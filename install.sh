#!/usr/bin/env bash
# Author: Kent Knox

# #################################################
# Pre-requisites check
# #################################################
# Exit code 0: alls well
# Exit code 1: problems with getopt
# Exit code 2: problems with supported platforms

# check if getopt command is installed
type getopt > /dev/null
if [[ $? -ne 0 ]]; then
  echo "This script uses getopt to parse arguments; try installing the util-linux package";
  exit 1
fi

# lsb-release file describes the system
if [[ ! -e "/etc/lsb-release" ]]; then
  echo "This script depends on the /etc/lsb-release file"
  exit 2
fi
source /etc/lsb-release

if [[ ${DISTRIB_ID} != Ubuntu ]]; then
  echo "This script only validated with Ubuntu"
  exit 2
fi

# #################################################
# helper functions
# #################################################
function display_help()
{
  echo "rocsparse build & installation helper script"
  echo "./install [-h|--help] "
  echo "    [-h|--help] prints this help message"
  echo "    [-i|--install] install after build"
  echo "    [-d|--dependencies] install build dependencies"
  echo "    [-c|--clients] build library clients too (combines with -i & -d)"
  echo "    [-g|--debug] -DCMAKE_BUILD_TYPE=Debug (default is =Release)"
  echo "    [--hip] build library for hip backend (default)"
  echo "    [--host] build library for host backend only"
  echo "    [--cuda] build library for cuda backend"
}

# This function is helpful for dockerfiles that do not have sudo installed, but the default user is root
elevate_if_not_root( )
{
  local uid=$(id -u)

  if (( ${uid} )); then
    sudo $@
  else
    $@
  fi
}

# #################################################
# global variables
# #################################################
install_package=false
install_dependencies=false
build_clients=false
build_hip=true
build_cuda=false
build_release=true

# #################################################
# Parameter parsing
# #################################################

# check if we have a modern version of getopt that can handle whitespace and long parameters
getopt -T
if [[ $? -eq 4 ]]; then
  GETOPT_PARSE=$(getopt --name "${0}" --longoptions help,install,clients,dependencies,debug,hip,host,cuda --options hicgd -- "$@")
else
  echo "Need a new version of getopt"
  exit 1
fi

if [[ $? -ne 0 ]]; then
  echo "getopt invocation failed; could not parse the command line";
  exit 1
fi

eval set -- "${GETOPT_PARSE}"

while true; do
  case "${1}" in
    -h|--help)
        display_help
        exit 0
        ;;
    -i|--install)
        install_package=true
        shift ;;
    -d|--dependencies)
        install_dependencies=true
        shift ;;
    -c|--clients)
        build_clients=true
        shift ;;
    -g|--debug)
        build_release=false
        shift ;;
    --hip)
        build_hip=true
        shift ;;
    --host)
        build_host=true
        build_hip=false
        shift ;;
    --cuda)
        build_cuda=true
        shift ;;
    --) shift ; break ;;
    *)  echo "Unexpected command line parameter received; aborting";
        exit 1
        ;;
  esac
done

build_dir=./build
printf "\033[32mCreating project build directory in: \033[33m${build_dir}\033[0m\n"

# #################################################
# prep
# #################################################
# ensure a clean build environment
if [[ "${build_release}" == true ]]; then
  rm -rf ${build_dir}/release
else
  rm -rf ${build_dir}/debug
fi

# #################################################
# install build dependencies on request
# #################################################
if [[ "${install_dependencies}" == true ]]; then
  # dependencies needed for rocsparse and clients to build
  library_dependencies_ubuntu=( "make" "cmake-curses-gui" "pkg-config" )

  if [[ "${build_hip}" == true ]]; then
    library_dependencies_ubuntu+=( "hip_hcc" "hcc" )
  fi

  if [[ "${build_cuda}" == true ]]; then
    library_dependencies_ubuntu+=( "hip_hcc" "cuda" )
  fi

  client_dependencies_ubuntu=( "libboost-program-options-dev" )

  elevate_if_not_root apt update

  # Dependencies required by main library
  for package in "${library_dependencies_ubuntu[@]}"; do
    if [[ $(dpkg-query --show --showformat='${db:Status-Abbrev}\n' ${package} 2> /dev/null | grep -q "ii"; echo $?) -ne 0 ]]; then
      printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
      elevate_if_not_root apt install -y --no-install-recommends ${package}
    fi
  done

  # Dependencies required by library client apps
  if [[ "${build_clients}" == true ]]; then
    for package in "${client_dependencies_ubuntu[@]}"; do
      if [[ $(dpkg-query --show --showformat='${db:Status-Abbrev}\n' ${package} 2> /dev/null | grep -q "ii"; echo $?) -ne 0 ]]; then
        printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
        elevate_if_not_root apt install -y --no-install-recommends ${package}
      fi
    done

    # The following builds googletest from source
    pushd .
      printf "\033[32mBuilding \033[33mgoogletest\033[32m from source"
      mkdir -p ${build_dir}/deps && cd ${build_dir}/deps
      cmake -DBUILD_BOOST=OFF -DCMAKE_INSTALL_PREFIX=deps-install ../../deps
      make -j$(nproc)
      # elevate_if_not_root make install
      make install
    popd
  fi

fi

export PATH=${PATH}:/opt/rocm/bin

pushd .
  # #################################################
  # configure & build
  # #################################################
  cmake_common_options=""
  cmake_client_options=""

  # build type
  if [[ "${build_release}" == true ]]; then
    mkdir -p ${build_dir}/release/clients && cd ${build_dir}/release
    cmake_common_options="${cmake_common_options} -DCMAKE_BUILD_TYPE=Release"
  else
    mkdir -p ${build_dir}/debug/clients && cd ${build_dir}/debug
    cmake_common_options="${cmake_common_options} -DCMAKE_BUILD_TYPE=Debug"
  fi

  # clients
  if [[ "${build_clients}" == true ]]; then
    cmake_client_options="${cmake_client_options} -DBUILD_CLIENTS_SAMPLES=ON -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON"
  fi

  if [[ "${build_host}" == true ]]; then
    cmake ${cmake_common_options} ${cmake_client_options} -DSUPPORT_HIP=OFF -DSUPPORT_OMP=OFF -DSUPPORT_MPI=OFF -DCMAKE_PREFIX_PATH="$(pwd)/../deps/deps-install" ../..
  fi

  if [[ "${build_hip}" == true ]]; then
    cmake ${cmake_common_options} ${cmake_client_options} -DSUPPORT_HIP=ON -DSUPPORT_OMP=OFF -DSUPPORT_MPI=OFF -DCMAKE_PREFIX_PATH="$(pwd)/../deps/deps-install" ../..
  fi

  if [[ "${build_cuda}" == true ]]; then
    cmake ${cmake_common_options} ${cmake_client_options} -DSUPPORT_HIP=ON -DSUPPORT_OMP=OFF -DSUPPORT_MPI=OFF -DCMAKE_PREFIX_PATH="$(pwd)/../deps/deps-install" ../..
  fi

  make -j$(nproc)

  # #################################################
  # install
  # #################################################
  # installing through package manager, which makes uninstalling easy
  if [[ "${install_package}" == true ]]; then
    make package
    elevate_if_not_root dpkg -i rocsparse-*.deb
  fi
popd
