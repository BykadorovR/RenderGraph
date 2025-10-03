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

void Shader::add(const std::vector<char>& shaderCode, VkShaderStageFlagBits type) {
  VkShaderModule shaderModule = _createShaderModule(shaderCode);
  _shaders[type] = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = type,
                    .module = shaderModule,
                    .pName = "main"};
}

void Shader::setSpecializationInfo(const VkSpecializationInfo& info, VkShaderStageFlagBits type) {
  _specializationInfo = info;
  _shaders[type].pSpecializationInfo = &_specializationInfo;
}

const VkPipelineShaderStageCreateInfo& Shader::getShaderStageInfo(VkShaderStageFlagBits type) const noexcept {
  return _shaders.at(type);
}

Shader::~Shader() {
  for (auto&& [type, shader] : _shaders) vkDestroyShaderModule(_device->getLogicalDevice(), shader.module, nullptr);
}