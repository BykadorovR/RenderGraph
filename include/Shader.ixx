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
  std::vector<SpvReflectShaderModule> _modules;
  std::vector<SpvReflectInterfaceVariable*> _variables;
  std::vector<VkDescriptorSetLayoutBinding> _descriptorSetLayoutBindings;
  std::vector<VkVertexInputAttributeDescription> _vertexInputAttributes;
  std::vector<VkVertexInputBindingDescription> _bindingDescription;
  std::unique_ptr<VkPipelineVertexInputStateCreateInfo> _vertexInputInfo;
  VkShaderModule _createShaderModule(const std::vector<char>& code);
  std::vector<VkVertexInputAttributeDescription> _calculateAttributeDescription(const SpvReflectInterfaceVariable* v,
                                                                                int binding,
                                                                                uint32_t& offset);

 public:
  Shader(const Device& device) noexcept;
  Shader(const Shader&) = delete;
  Shader& operator=(const Shader&) = delete;
  Shader(Shader&&) = delete;
  Shader& operator=(Shader&&) = delete;

  void add(const std::vector<char>& shaderCode, const VkSpecializationInfo* info = nullptr);
  // returns info about shader stages (vertex, tesselation, fragments), used during pipeline creation
  std::vector<VkPipelineShaderStageCreateInfo> getShaderStageInfo() const noexcept;
  // used to create descriptor set layout (provides all neccessary info about bindings and types)
  const std::vector<VkDescriptorSetLayoutBinding>& getDescriptorSetLayoutBindings() const;
  // for instancing
  const VkPipelineVertexInputStateCreateInfo* getVertexInputInfo(
      std::vector<std::pair<VkVertexInputRate, int>> typeElements);
  // provides information about attributes and their format (position, UV) + (RGBA32, RG32), used during pipeline creation
  const VkPipelineVertexInputStateCreateInfo* getVertexInputInfo();
  ~Shader();
};
}  // namespace RenderGraph