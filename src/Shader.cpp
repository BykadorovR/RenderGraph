module Shader;
import <ranges>;
import <algorithm>;
using namespace RenderGraph;

VkShaderModule Shader::_createShaderModule(const std::vector<char>& code) {
  VkShaderModuleCreateInfo createInfo{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = code.size(),
      .pCode = reinterpret_cast<const uint32_t*>(code.data()),
  };

  VkShaderModule shaderModule;
  if (vkCreateShaderModule(_device->getLogicalDevice(), &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
    throw std::runtime_error("failed to create shader module!");
  }

  return shaderModule;
}

std::vector<VkVertexInputAttributeDescription>
Shader::_calculateAttributeDescription(const SpvReflectInterfaceVariable* v, int binding, uint32_t& offset) {
  const uint32_t columnCount = std::max(v->numeric.matrix.column_count, 1u);
  const uint32_t componentCount = v->numeric.matrix.column_count > 0 ? v->numeric.matrix.row_count
                                                                     : std::max(v->numeric.vector.component_count, 1u);

  const uint32_t columnSize = (v->numeric.scalar.width / 8) * componentCount;
  std::vector<VkVertexInputAttributeDescription> attributes;
  attributes.reserve(columnCount);
  for (uint32_t column = 0; column < columnCount; ++column) {
    attributes.push_back({
        .location = v->location + column,
        .binding = static_cast<uint32_t>(binding),
        .format = static_cast<VkFormat>(v->format),
        .offset = offset,
    });

    offset += columnSize;
  }

  return attributes;
}

Shader::Shader(const Device& device) noexcept : _device(&device) {}

void Shader::add(const std::vector<char>& shaderCode, const VkSpecializationInfo* info) {
  // parse spirv code
  SpvReflectShaderModule module;
  SpvReflectResult r = spvReflectCreateShaderModule(shaderCode.size(), shaderCode.data(), &module);
  if (r != SPV_REFLECT_RESULT_SUCCESS) {
    throw std::runtime_error("Failed to reflect shader module");
  }

  uint32_t descriptorCount = 0;
  SpvReflectResult result = spvReflectEnumerateDescriptorSets(&module, &descriptorCount, nullptr);
  if (result != SPV_REFLECT_RESULT_SUCCESS) {
    spvReflectDestroyShaderModule(&module);
    throw std::runtime_error("Failed to enumerate descriptor sets");
  }

  std::vector<SpvReflectDescriptorSet*> sets(descriptorCount);
  result = spvReflectEnumerateDescriptorSets(&module, &descriptorCount, sets.data());
  if (result != SPV_REFLECT_RESULT_SUCCESS) {
    spvReflectDestroyShaderModule(&module);
    throw std::runtime_error("Failed to enumerate descriptor sets");
  }

  for (const SpvReflectDescriptorSet* reflectedSet : sets) {
    const uint32_t setIndex = reflectedSet->set;

    if (_descriptorSetLayoutBindings.size() <= setIndex) {
      _descriptorSetLayoutBindings.resize(setIndex + 1);
    }

    auto& destinationBindings = _descriptorSetLayoutBindings[setIndex];
    for (uint32_t i = 0; i < reflectedSet->binding_count; ++i) {
      const SpvReflectDescriptorBinding* reflectedBinding = reflectedSet->bindings[i];
      VkDescriptorSetLayoutBinding layoutBinding{
          .binding = reflectedBinding->binding,
          .descriptorType = static_cast<VkDescriptorType>(reflectedBinding->descriptor_type),
          .descriptorCount = reflectedBinding->count,
          .stageFlags = static_cast<VkShaderStageFlags>(module.shader_stage),
          .pImmutableSamplers = nullptr,
      };

      auto position = std::lower_bound(
          destinationBindings.begin(), destinationBindings.end(), layoutBinding.binding,
          [](const VkDescriptorSetLayoutBinding& existing, uint32_t binding) { return existing.binding < binding; });

      if (position != destinationBindings.end() && position->binding == layoutBinding.binding) {
        // same set/binding can be used by multiple stages, need to create one record with merged stageFlags
        if (position->descriptorType != layoutBinding.descriptorType ||
            position->descriptorCount != layoutBinding.descriptorCount) {
          throw std::runtime_error("Incompatible descriptor declarations for the same set and binding");
        }
        position->stageFlags |= layoutBinding.stageFlags;
      } else {
        destinationBindings.insert(position, layoutBinding);
      }
    }
  }
  VkShaderModule shaderModule = _createShaderModule(shaderCode);
  _shaders[static_cast<VkShaderStageFlagBits>(module.shader_stage)] = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = static_cast<VkShaderStageFlagBits>(module.shader_stage),
      .module = shaderModule,
      .pName = "main",
      .pSpecializationInfo = info,
  };
  _specializationInfo[static_cast<VkShaderStageFlagBits>(module.shader_stage)] = info;
  if (module.shader_stage == SPV_REFLECT_SHADER_STAGE_VERTEX_BIT) {
    uint32_t variableCount = 0;
    SpvReflectResult result = spvReflectEnumerateInputVariables(&module, &variableCount, nullptr);
    if (result != SPV_REFLECT_RESULT_SUCCESS) {
      spvReflectDestroyShaderModule(&module);
      throw std::runtime_error("Failed to enumerate vertex input variables");
    }
    std::vector<SpvReflectInterfaceVariable*> variables(variableCount);
    result = spvReflectEnumerateInputVariables(&module, &variableCount, variables.data());
    if (result != SPV_REFLECT_RESULT_SUCCESS) {
      spvReflectDestroyShaderModule(&module);
      throw std::runtime_error("Failed to enumerate vertex input variables");
    }
    variables.resize(variableCount);
    std::erase_if(variables, [](const SpvReflectInterfaceVariable* variable) {
      return variable->decoration_flags & SPV_REFLECT_DECORATION_BUILT_IN;
    });
    std::ranges::sort(variables, {}, &SpvReflectInterfaceVariable::location);
    _variables = std::move(variables);
  }

  _modules.push_back(module);
}

std::vector<VkPipelineShaderStageCreateInfo> Shader::getShaderStageInfo() const noexcept {
  return _shaders | std::views::values | std::ranges::to<std::vector>();
}

const std::vector<std::vector<VkDescriptorSetLayoutBinding>>& Shader::getDescriptorSetLayoutBindings() const {
  return _descriptorSetLayoutBindings;
}

const VkPipelineVertexInputStateCreateInfo* Shader::getVertexInputInfo() {
  if (_vertexInputInfo == nullptr) {
    uint32_t attributesSize = 0;
    for (auto v : _variables) {
      auto attributes = _calculateAttributeDescription(v, 0, attributesSize);

      _vertexInputAttributes.insert(_vertexInputAttributes.end(), attributes.begin(), attributes.end());
    }
    if (!_vertexInputAttributes.empty()) {
      _bindingDescription = {{
          .binding = 0,
          .stride = attributesSize,
          .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
      }};
    }
    _vertexInputInfo = std::make_unique<VkPipelineVertexInputStateCreateInfo>(
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, nullptr, 0,
        static_cast<uint32_t>(_bindingDescription.size()), _bindingDescription.data(),
        static_cast<uint32_t>(_vertexInputAttributes.size()), _vertexInputAttributes.data());
  }

  return _vertexInputInfo.get();
}

const VkPipelineVertexInputStateCreateInfo* Shader::getVertexInputInfo(
    std::vector<std::pair<VkVertexInputRate, int>> typeElements) {
  if (_vertexInputInfo == nullptr) {
    _bindingDescription.resize(typeElements.size());
    int locationOffset = 0;
    for (int binding = 0; binding < typeElements.size(); ++binding) {
      auto [type, number] = typeElements[binding];
      uint32_t offset = 0;
      for (int location = locationOffset; location < locationOffset + number; ++location) {
        auto v = _variables[location];
        auto attributes = _calculateAttributeDescription(v, binding, offset);
        _vertexInputAttributes.insert(_vertexInputAttributes.end(), attributes.begin(), attributes.end());
      }
      locationOffset += number;
      _bindingDescription[binding] = VkVertexInputBindingDescription{
          .binding = static_cast<uint32_t>(binding),
          .stride = offset,
          .inputRate = type,
      };
    }

    _vertexInputInfo = std::make_unique<VkPipelineVertexInputStateCreateInfo>(
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, nullptr, 0,
        static_cast<uint32_t>(_bindingDescription.size()), _bindingDescription.data(),
        static_cast<uint32_t>(_vertexInputAttributes.size()), _vertexInputAttributes.data());
  }

  return _vertexInputInfo.get();
}

Shader::~Shader() {
  for (auto&& [type, shader] : _shaders) {
    vkDestroyShaderModule(_device->getLogicalDevice(), shader.module, nullptr);
  }

  for (auto& module : _modules) {
    spvReflectDestroyShaderModule(&module);
  }
}