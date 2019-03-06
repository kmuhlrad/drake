#include "drake/systems/framework/system.h"

#include <iomanip>
#include <ios>
#include <regex>

namespace drake {
namespace systems {

std::string SystemImpl::GetMemoryObjectName(
    const std::string& nice_type_name, int64_t address) {
  std::cout << "trying to get memory object name" << std::endl;
  using std::setfill;
  using std::setw;
  using std::hex;

  // Remove the template parameter(s).
  const std::string type_name_without_templates = std::regex_replace(
      nice_type_name, std::regex("<.*>$"), std::string());
  std::cout << "Removed template params" << std::endl;

  // Replace "::" with "/" because ":" is System::GetSystemPathname's separator.
  // TODO(sherm1) Change the separator to "/" and avoid this!
  const std::string default_name = std::regex_replace(
      type_name_without_templates, std::regex(":+"), std::string("/"));
  std::cout << "replaced ::" << std::endl;

  // Append the address spelled like "@0123456789abcdef".
  std::ostringstream result;
  result << default_name << '@' << setfill('0') << setw(16) << hex << address;
  std::cout << "wrote output" << std::endl;
  return result.str();
}

}  // namespace systems
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::systems::System)
