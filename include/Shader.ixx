export module Shader;
import Device;
import <volk.h>;
import <unordered_map>;
import <vector>;
import <memory>;
import <spirv_reflect.h>;

export namespace RenderGraph {
class Shader final {
 private:
  const Device* _device;
  std::unordered_map<VkShaderStageFlagBits, VkPipelineShaderStageCreateInfo> _shaders;
  std::unordered_map<VkShaderStageFlagBits, const VkSpecializationInfo*> _specializationInfo;
  std::vector<VkDescriptorSetLayoutBinding> _descriptorSetLayoutBindings;
  std::vector<VkVertexInputAttributeDescription> _vertexInputAttributes;
  std::vector<VkVertexInputBindingDescription> _bindingDescription;
  std::unique_ptr<VkPipelineVertexInputStateCreateInfo> _vertexInputInfo;
  uint32_t _attributesSize = 0;
  VkShaderModule _createShaderModule(const std::vector<char>& code);
 public:
  Shader(const Device& device) noexcept;
  Shader(const Shader&) = delete;
  Shader& operator=(const Shader&) = delete;
  Shader(Shader&&) = delete;
  Shader& operator=(Shader&&) = delete;
  
  void add(const std::vector<char>& shaderCode, const VkSpecializationInfo* info = nullptr);
  std::vector<VkPipelineShaderStageCreateInfo> getShaderStageInfo() const noexcept;
  const std::vector<VkDescriptorSetLayoutBinding>& getDescriptorSetLayoutBindings() const;
  // for instancing
  const VkPipelineVertexInputStateCreateInfo* getVertexInputInfo(
      std::vector<std::pair<VkVertexInputRate, int>> typeSize);
  const VkPipelineVertexInputStateCreateInfo* getVertexInputInfo();
  ~Shader();
};
}  // namespace RenderGraph