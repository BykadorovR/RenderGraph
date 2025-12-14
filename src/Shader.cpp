module Shader;
using namespace RenderGraph;

VkShaderModule Shader::_createShaderModule(const std::vector<char>& code) {
  VkShaderModuleCreateInfo createInfo{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                      .codeSize = code.size(),
                                      .pCode = reinterpret_cast<const uint32_t*>(code.data())};

  VkShaderModule shaderModule;
  if (vkCreateShaderModule(_device->getLogicalDevice(), &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
    throw std::runtime_error("failed to create shader module!");
  }

  return shaderModule;
}

Shader::Shader(const Device& device) noexcept : _device(&device) {}

void Shader::add(const std::vector<char>& shaderCode, const VkSpecializationInfo* info) {
  // parse spirv code
  SpvReflectShaderModule module;
  SpvReflectResult r = spvReflectCreateShaderModule(shaderCode.size() * sizeof(uint32_t), shaderCode.data(), &module);
  if (r != SPV_REFLECT_RESULT_SUCCESS) {
    throw std::runtime_error("Failed to reflect shader module");
  }
  uint32_t count = 0;
  spvReflectEnumerateDescriptorBindings(&module, &count, nullptr);
  std::vector<SpvReflectDescriptorBinding*> bindings(count);
  spvReflectEnumerateDescriptorBindings(&module, &count, bindings.data());

  for (uint32_t i = 0; i < count; ++i) {
    SpvReflectDescriptorBinding* b = bindings[i];
    VkDescriptorSetLayoutBinding layoutBinding{};
    layoutBinding.binding = b->binding;
    layoutBinding.descriptorType = static_cast<VkDescriptorType>(b->descriptor_type);
    layoutBinding.descriptorCount = b->count;
    layoutBinding.stageFlags = static_cast<VkShaderStageFlagBits>(module.shader_stage);
    layoutBinding.pImmutableSamplers = nullptr;

    auto it = std::lower_bound(_descriptorSetLayoutBindings.begin(), _descriptorSetLayoutBindings.end(), layoutBinding,
                               [](auto const& x, auto const& v) { return x.binding < v.binding; });
    _descriptorSetLayoutBindings.insert(it, layoutBinding);
  }

  VkShaderModule shaderModule = _createShaderModule(shaderCode);
  _shaders[static_cast<VkShaderStageFlagBits>(module.shader_stage)] = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = static_cast<VkShaderStageFlagBits>(module.shader_stage),
      .module = shaderModule,
      .pName = "main",
      .pSpecializationInfo = info};

  _specializationInfo[static_cast<VkShaderStageFlagBits>(module.shader_stage)] = info;
}

const VkPipelineShaderStageCreateInfo& Shader::getShaderStageInfo(VkShaderStageFlagBits type) const noexcept {
  return _shaders.at(type);
}

const std::vector<VkDescriptorSetLayoutBinding>& Shader::getDescriptorSetLayoutBindings() const {
  return _descriptorSetLayoutBindings;
}

Shader::~Shader() {
  for (auto&& [type, shader] : _shaders) vkDestroyShaderModule(_device->getLogicalDevice(), shader.module, nullptr);
}