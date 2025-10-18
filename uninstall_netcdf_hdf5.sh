#!/bin/bash
# Function to remove netcdf and hdf5 libraries from the system
# Created by: Kushagra Srivastava
# Date: 2025-02-27

# Runs in sudo: be careful

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    case $color in
        "green") echo -e "\033[0;32m$message\033[0m" ;;
        "red") echo -e "\033[0;31m$message\033[0m" ;;
        "yellow") echo -e "\033[1;33m$message\033[0m" ;;
    esac
}

# Function to safely remove files and directories
safe_remove() {
    local path=$1
    if [ -e "$path" ]; then
        sudo rm -rf "$path"
        if [ $? -eq 0 ]; then
            print_status "green" "Successfully removed: $path"
        else
            print_status "red" "Failed to remove: $path"
        fi
    fi
}

print_status "yellow" "Starting uninstallation of NetCDF and HDF5 libraries..."
print_status "yellow" "This will remove these libraries from /usr/local"
echo
read -p "Do you want to continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    print_status "yellow" "Uninstallation cancelled."
    exit 1
fi

# Array of paths to remove
declare -a paths=(
    # NetCDF-CXX4
    "/usr/local/lib/libnetcdf-cxx4*"
    "/usr/local/lib/libnetcdf_c++*"
    "/usr/local/include/netcdf"
    "/usr/local/lib/cmake/netCDF-cxx4"
    
    # NetCDF-C
    "/usr/local/lib/libnetcdf*"
    "/usr/local/include/netcdf*"
    "/usr/local/lib/cmake/netCDF"
    
    # HDF5
    "/usr/local/lib/libhdf5*"
    "/usr/local/include/hdf5*"
    "/usr/local/lib/cmake/hdf5"
    
    # pkg-config files
    "/usr/local/lib/pkgconfig/netcdf*"
    "/usr/local/lib/pkgconfig/hdf5*"
    
    # CMake files
    "/usr/local/cmake/netcdf*"
    "/usr/local/cmake/hdf5*"
)

print_status "yellow" "Removing libraries..."
echo

# Remove each path
for path in "${paths[@]}"; do
    safe_remove "$path"
done

# Clean pkg-config cache
if command -v pkg-config >/dev/null 2>&1; then
    print_status "yellow" "Cleaning pkg-config cache..."
    sudo pkg-config --reconfigure
fi

# Clear CMake package registry
print_status "yellow" "Cleaning CMake package registry..."
rm -rf ~/.cmake/packages/*/NetCDF* 2>/dev/null
rm -rf ~/.cmake/packages/*/HDF5* 2>/dev/null

print_status "yellow" "Cleaning system cache..."
sudo ldconfig

print_status "green" "Uninstallation complete!"
print_status "yellow" "Note: You may need to restart your terminal or system for all changes to take effect." 