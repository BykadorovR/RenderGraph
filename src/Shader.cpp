module Shader;
import <ranges>;
import <algorithm>;
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
  SpvReflectResult r = spvReflectCreateShaderModule(shaderCode.size(), shaderCode.data(), &module);
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

  if (module.shader_stage == SPV_REFLECT_SHADER_STAGE_VERTEX_BIT) {
    count = 0;
    spvReflectEnumerateInputVariables(&module, &count, nullptr);
    std::vector<SpvReflectInterfaceVariable*> vars(count);
    spvReflectEnumerateInputVariables(&module, &count, vars.data());
    
    std::sort(vars.begin(), vars.end(),
              [&](const SpvReflectInterfaceVariable* left, const SpvReflectInterfaceVariable* right) {
                return left->location < right->location;
              });
    
    for (auto v : vars) {        
      if (v->decoration_flags & SPV_REFLECT_DECORATION_BUILT_IN) continue;  // skip builtins      
      uint32_t loc = v->location;
      VkFormat fmt = static_cast<VkFormat>(v->format);

      auto elements = v->numeric.matrix.column_count * v->numeric.matrix.row_count;
      elements = std::max(elements, v->numeric.vector.component_count);
      elements = std::max(elements, 1u);

      // split attributes larger than vec4
      std::vector<int> components;
      while ((elements / 4) > 0) {
        components.push_back(4);
        elements -= 4;
      }
      if (elements) components.push_back(elements % 4);
      
      for (auto component : components) {
        auto size = (v->numeric.scalar.width / 8) * component;

        VkVertexInputAttributeDescription attributes{};
        attributes.location = loc;
        attributes.format = fmt;
        attributes.binding = 0;
        attributes.offset = _attributesSize;
        _vertexInputAttributes.push_back(attributes);
        
        loc++;
        _attributesSize += size;
      }
    }    
  }
  spvReflectDestroyShaderModule(&module);
}

std::vector<VkPipelineShaderStageCreateInfo> Shader::getShaderStageInfo() const noexcept {
  return _shaders | std::views::values | std::ranges::to<std::vector>();
}

const std::vector<VkDescriptorSetLayoutBinding>& Shader::getDescriptorSetLayoutBindings() const {
  return _descriptorSetLayoutBindings;
}

const VkPipelineVertexInputStateCreateInfo* Shader::getVertexInputInfo() {
  if (_vertexInputInfo == nullptr) {
    if (_vertexInputAttributes.size() > 0) {
      _bindingDescription = {{
          .binding = 0,
          .stride = _attributesSize,
          .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
      }};
    }

    _vertexInputInfo = std::make_unique<VkPipelineVertexInputStateCreateInfo>(
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, nullptr, 0, _bindingDescription.size(),
        _bindingDescription.data(), static_cast<uint32_t>(_vertexInputAttributes.size()),
        _vertexInputAttributes.data());
  }

  return _vertexInputInfo.get();
}

const VkPipelineVertexInputStateCreateInfo* Shader::getVertexInputInfo(
    std::vector<std::pair<VkVertexInputRate, int>> typeSize) {
  if (_vertexInputInfo == nullptr) {
    _bindingDescription.resize(typeSize.size());
    int index = 0;
    for (int binding = 0; binding < typeSize.size(); binding++) {
      auto [type, size] = typeSize[binding];
      size += _vertexInputAttributes[index].offset;

      while (index < _vertexInputAttributes.size() && _vertexInputAttributes[index].offset < size) {
        _vertexInputAttributes[index].binding = binding;
        index++;        
      }

      uint32_t stride = _attributesSize;
      if (index < _vertexInputAttributes.size()) stride = _vertexInputAttributes[index].offset;
      _bindingDescription[binding] = VkVertexInputBindingDescription{.binding = static_cast<uint32_t>(binding),
                                                                     .stride = stride,
                                                                     .inputRate = type};
    }

    _vertexInputInfo = std::make_unique<VkPipelineVertexInputStateCreateInfo>(
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, nullptr, 0, _bindingDescription.size(),
        _bindingDescription.data(), static_cast<uint32_t>(_vertexInputAttributes.size()),
        _vertexInputAttributes.data());
  }
  return _vertexInputInfo.get();
}

Shader::~Shader() {
  for (auto&& [type, shader] : _shaders) vkDestroyShaderModule(_device->getLogicalDevice(), shader.module, nullptr);
}