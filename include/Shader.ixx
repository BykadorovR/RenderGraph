export module Shader;
import Device;
import <volk.h>;
import <map>;
import <vector>;

export namespace RenderGraph {
class Shader final {
 private:
  const Device* _device;
  std::map<VkShaderStageFlagBits, VkPipelineShaderStageCreateInfo> _shaders;
  VkSpecializationInfo _specializationInfo;
  VkShaderModule _createShaderModule(const std::vector<char>& code);

 public:
  Shader(const Device& device) noexcept;
  Shader(const Shader&) = delete;
  Shader& operator=(const Shader&) = delete;
  Shader(Shader&&) = delete;
  Shader& operator=(Shader&&) = delete;

  void add(const std::vector<char>& shaderCode, VkShaderStageFlagBits type);
  void setSpecializationInfo(const VkSpecializationInfo& info, VkShaderStageFlagBits type);
  const VkPipelineShaderStageCreateInfo& getShaderStageInfo(VkShaderStageFlagBits type) const noexcept;
  ~Shader();
};
}  // namespace RenderGraph