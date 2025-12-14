export module Shader;
import Device;
import <volk.h>;
import <map>;
import <vector>;
import <spirv_reflect.h>;

export namespace RenderGraph {
class Shader final {
 private:
  const Device* _device;
  std::map<VkShaderStageFlagBits, VkPipelineShaderStageCreateInfo> _shaders;
  std::map < VkShaderStageFlagBits, const VkSpecializationInfo*> _specializationInfo;
  std::vector<VkDescriptorSetLayoutBinding> _descriptorSetLayoutBindings;
  VkShaderModule _createShaderModule(const std::vector<char>& code);
 public:
  Shader(const Device& device) noexcept;
  Shader(const Shader&) = delete;
  Shader& operator=(const Shader&) = delete;
  Shader(Shader&&) = delete;
  Shader& operator=(Shader&&) = delete;
  
  void add(const std::vector<char>& shaderCode, const VkSpecializationInfo* info = nullptr);
  const VkPipelineShaderStageCreateInfo& getShaderStageInfo(VkShaderStageFlagBits type) const noexcept;
  const std::vector<VkDescriptorSetLayoutBinding>& getDescriptorSetLayoutBindings() const;  
  ~Shader();
};
}  // namespace RenderGraph